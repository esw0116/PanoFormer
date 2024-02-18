from __future__ import absolute_import, division, print_function
import os, glob
import argparse
from PIL import Image
import numpy as np
import open3d as o3d
import cv2

from testers2d3d import Trainer

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
    trainer = Trainer(args)
    input_files = sorted(glob.glob(os.path.join(args.input_folder, '*.png')))
    for input_file in input_files:
        dir_name, img_name, img_ext = os.path.dirname(input_file), os.path.splitext(os.path.basename(input_file))[0], os.path.splitext(os.path.basename(input_file))[1]
        print(img_name)
        if img_name.endswith('_mask'):
            continue

        mask_file = os.path.join(dir_name, img_name+'_mask'+img_ext)

        output_depth, output_depth_colorized, mask_flag = trainer.process_batch(input_file, mask_file)
        if mask_flag:
            img_name = img_name + '_withmask'
        Image.fromarray(output_depth_colorized).save(os.path.join('outputs', img_name+img_ext))

        # Unproject to point cloud
        img = cv2.imread(input_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W = output_depth.shape[-2:]
        depth = output_depth.squeeze(0).squeeze(0).numpy()
        equi_x, equi_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        # Move the center of image
        moved_x = equi_x - W / 2 + 0.5
        moved_y = (-1) * equi_y + H / 2 + 0.5
        # Normalize coordinates
        norm_x = moved_x / (W / 2 - 0.5)
        norm_y = moved_y / (H / 2 - 0.5)
        # Calculate longitude and latitude
        longitude = norm_x * np.pi
        latitude = norm_y * np.pi / 2

        x = np.cos(latitude) * np.sin(longitude) * depth
        y = np.cos(latitude) * np.cos(longitude) * depth
        z = np.sin(latitude) * depth
        
        point = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1)
        color = img.reshape(-1, 3).astype(np.float32)/255.
        pcd_o3d = o3d.geometry.PointCloud()

        pcd_o3d.points = o3d.utility.Vector3dVector(point)
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(os.path.join('outputs', img_name+'.ply'), pcd_o3d)
        breakpoint()


if __name__ == "__main__":
    main()
