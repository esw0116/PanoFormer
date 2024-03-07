import os, glob
import argparse
from PIL import Image
import numpy as np
import open3d as o3d
import cv2
import imageio
import json

import cam_utils
from testers2d3d import Trainer
from utils import colorize, Marigold_estimation
from Marigold.marigold_pipeline import MarigoldPipeline


parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Training")

# system settings
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")

# model settings
parser.add_argument("--model_name", type=str, default="panodepth", help="folder to save the model in")
parser.add_argument("--input_folder", type=str, default="inputs", help="folder to save the model in")

# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")

# loading and logging settings
parser.add_argument("--load_weights_dir", default='./tmp_s2d3d/panodepth/models/weights', type=str, help="folder of model to load")#, default='./tmp_abl_offset/panodepth/models/weights_49'
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "tmp_s2d3dtest"), help="log directory")
parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")

# data augmentation settings
parser.add_argument("--disable_color_augmentation", action="store_true", help="if set, do not use color augmentation")
parser.add_argument("--disable_LR_filp_augmentation", action="store_true",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", action="store_true",
                    help="if set, do not use yaw rotation augmentation")

args = parser.parse_args()



def main():
    marigold_model = MarigoldPipeline.from_pretrained('tmp_s2d3d/Marigold_v1_merged').to('cuda')

    trainer = Trainer(args)
    input_files = sorted(glob.glob(os.path.join(args.input_folder, '*.mp4')))
    for input_file in input_files:
        dir_name, vid_name = os.path.dirname(input_file), os.path.splitext(os.path.basename(input_file))[0]
        print(vid_name)

        # Create output folder
        if not os.path.exists(os.path.join('outputs_video', vid_name)):
            os.makedirs(os.path.join('outputs_video', vid_name))
        else:
            print(f'image folders for {vid_name} already exists!')
            continue

        vid_frames = imageio.mimread(input_file)
        vid_length = len(vid_frames)

        for i, img in enumerate(vid_frames):
            img = cv2.resize(img, (1024, 512))
            H, W = img.shape[:2]

            if not os.path.exists(os.path.join('outputs_video', vid_name, f'{i:03d}')):
                os.mkdir(os.path.join('outputs_video', vid_name, f'{i:03d}'))

            mask = None
            depth_mari = Marigold_estimation(marigold_model, Image.fromarray(img))
            if i == 0:
                depth_pano = trainer.process_batch(img, mask)
                depth_mari_low, depth_mari_high = np.quantile(depth_mari, 0.15), np.quantile(depth_mari, 0.85)
                depth_pano_low, depth_pano_high = np.quantile(depth_pano, 0.15), np.quantile(depth_pano, 0.85)
            depth_mari = (depth_pano_high - depth_pano_low) / (depth_mari_high - depth_mari_low) * (depth_mari - depth_mari_low) + depth_pano_low
            depth_colorized = colorize(depth_mari)

            Image.fromarray(img).save(os.path.join('outputs_video', vid_name, f'{i:03d}', 'image00.png'))
            Image.fromarray(depth_colorized).save(os.path.join('outputs_video', vid_name, f'{i:03d}', 'depth00.png'))

            # Unproject to point cloud
            scale = H / 512
            depth = depth_mari * scale
            np.save(os.path.join('outputs_video', vid_name, f'{i:03d}', 'depth00.npy'), depth)

            # intrinsic matrix K (from phi theta to u v)
            K = np.array([[(W/2 - 0.5)/np.pi, 0., W/2 - 0.5],
                [0., -2*(H/2 - 0.5)/np.pi, H/2 - 0.5],
                [0.,  0.,  1.]]).astype(np.float32)

            point = cam_utils.uv2sphere(depth, H, W, K).transpose((1,0))
            color = img.reshape(-1, 3).astype(np.float32)/255.
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(point)
            pcd_o3d.colors = o3d.utility.Vector3dVector(color)
            o3d.io.write_point_cloud(os.path.join('outputs_video', vid_name, f'{i:03d}', 'blender.ply'), pcd_o3d)


            blender_train_json = {}
            blender_train_json["camera_angle_x"] = 0.8279103882874479
            blender_train_json["frames"] = []
            w2c = np.identity(4)
            w2c[1,1] = -1
            w2c[2,2] = -1

            curr_frame = {}
            curr_frame["file_path"] = "image00"
            curr_frame["depth_path"] = "depth00"
            curr_frame["transform_matrix"] = w2c.tolist()
            (blender_train_json["frames"]).append(curr_frame)

            movement = np.array([[0.5,0,0], [-0.5,0,0], [0,0.5,0], [0,-0.5,0], [0,0,0.5], [0,0,-0.5]])
            for idx in range(6):
                mypoint = (point + movement[idx]).transpose((1,0))
                image, depth = cam_utils.sphere2uv(mypoint, color, H, W, K)
                Image.fromarray(image).save(os.path.join('outputs_video', vid_name, f'{i:03d}', f'image{idx+1:02d}.png'))
                np.save(os.path.join('outputs_video', vid_name, f'{i:03d}', f'depth{idx+1:02d}.npy'), depth)

                w2c = np.identity(4)
                w2c[1,1] = -1
                w2c[2,2] = -1
                w2c[:3, 3] = movement[idx] * np.array([1, -1, -1])

                curr_frame = {}
                curr_frame["file_path"] = "image{:02d}".format(idx+1)
                curr_frame["depth_path"] = "depth{:02d}".format(idx+1)
                curr_frame["transform_matrix"] = w2c.tolist()
                (blender_train_json["frames"]).append(curr_frame)

            train_json_path = os.path.join('outputs_video', vid_name, 'transforms_train.json')
            with open(train_json_path, 'w') as outfile:
                json.dump(blender_train_json, outfile, indent=4)

if __name__ == "__main__":
    main()
