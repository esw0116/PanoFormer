from __future__ import absolute_import, division, print_function
import os, glob
import argparse
from PIL import Image
import numpy as np
import open3d as o3d
import cv2

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
    input_files = sorted(glob.glob(os.path.join(args.input_folder, '*.png')) + glob.glob(os.path.join(args.input_folder, '*.jpg')))
    for input_file in input_files:
        dir_name, img_name, img_ext = os.path.dirname(input_file), os.path.splitext(os.path.basename(input_file))[0], os.path.splitext(os.path.basename(input_file))[1]
        mask_file = os.path.join(dir_name, img_name+'_mask'+img_ext)

        if img_name.endswith('_mask'):
            continue
        print(img_name)

        # Read image
        img = cv2.imread(input_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask
        try:
            mask = cv2.imread(mask_file)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            img_name = img_name + '_withmask'
        except:
            mask = None

        # Create output folder
        if not os.path.exists(os.path.join('outputs', 'Marigold', img_name)):
            os.makedirs(os.path.join('outputs', 'Marigold', img_name))

        depth_mari = Marigold_estimation(marigold_model, Image.fromarray(img))
        depth_pano = trainer.process_batch(img, mask)
        depth_mari_low, depth_mari_high = np.quantile(depth_mari, 0.15), np.quantile(depth_mari, 0.85)
        depth_pano_low, depth_pano_high = np.quantile(depth_pano, 0.15), np.quantile(depth_pano, 0.85)
        depth_mari = (depth_pano_high - depth_pano_low) / (depth_mari_high - depth_mari_low) * (depth_mari - depth_mari_low) + depth_pano_low
        depth_colorized = colorize(depth_mari)

        Image.fromarray(img).save(os.path.join('outputs', 'Marigold', img_name, img_name+'.png'))
        Image.fromarray(depth_colorized).save(os.path.join('outputs', 'Marigold', img_name, img_name+'_depth.png'))

        # Unproject to point cloud
        H, W = depth_mari.shape[-2:]
        scale = H / 512
        depth = depth_mari * scale

        if mask is None:
            mask = np.ones((H, W))
        else:
            mask = mask[..., :1]
        mask = mask.reshape(-1)

        # intrinsic matrix K (from phi theta to u v)
        K = np.array([[(W/2 - 0.5)/np.pi, 0., W/2 - 0.5],
            [0., -2*(H/2 - 0.5)/np.pi, H/2 - 0.5],
            [0.,  0.,  1.]]).astype(np.float32)

        point = cam_utils.uv2sphere(depth, H, W, K).transpose((1,0))
        point = point[np.where(mask)]
        point = point[:, [0,2,1]]
        point = point * np.array([[1,-1,1]])

        color = (img.reshape(-1, 3).astype(np.float32)/255.)[np.where(mask)]
        pcd_o3d = o3d.geometry.PointCloud()

        pcd_o3d.points = o3d.utility.Vector3dVector(point)
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(os.path.join('outputs', 'Marigold', img_name, img_name+'_pcd.ply'), pcd_o3d)

        # re-project image
        # re_point = point + np.stack([np.zeros(512*1024), np.ones(512*1024), np.zeros(512*1024)], axis=-1)
        # re_img, re_depth = cam_utils.sphere2erp(re_point.transpose(1,0), color.transpose(1,0), H, W, K)
        # Image.fromarray(re_img).save(os.path.join('outputs', 'Marigold', img_name, img_name+'_reproj.png'))
        # color_depth = colorize(re_depth)
        # Image.fromarray(color_depth).save(os.path.join('outputs', 'Marigold', img_name, img_name+'_redepth.png'))


if __name__ == "__main__":
    main()
