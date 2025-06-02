

"""
Refer to https://github.com/embodied-generalist/embodied-generalist/blob/main/data/datasets.py

scanrefer
- ScanRefer_filtered_train.json
- ScanRefer_filtered_val.json

scannet
- mask
- pcd_with_object_aabbs
- vg

"""

import os
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from scipy import sparse
from collections import defaultdict
from llava.video_utils import VideoProcessor
from llava.utils_3d import convert_pc_to_box
from llava.eval.box_utils import get_3d_box_corners, box3d_iou


def load_scene(filename):
    d = torch.load(filename)
    # return d['aabb_obj_ids'].tolist(), d['aabb_corner_xyz'].tolist()
    object_ids = d['aabb_obj_ids'].tolist()
    corner_xyz = d['aabb_corner_xyz'].tolist()

    ret = {}
    for i in range(len(object_ids)):
        object_id = str(object_ids[i])

        xs, ys, zs = zip(*corner_xyz[i])
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        l = z_max - z_min

        ret[object_id] = [x_center, y_center, z_center, w, h, l]

    return ret


def main(args):

    video_processor = VideoProcessor(
        voxel_size=args.voxel_size,
        min_xyz_range=None,
        max_xyz_range=None,
    )

    scan2pcd = {}

    # load ground truth box
    for split in ["train", "val", "test"]:
        for scene_id in tqdm(os.listdir(os.path.join(args.scannet_dir, 'pcd_with_object_aabbs', split))):
            scene_id = scene_id.split(".")[0]
            pcd_path = os.path.join(args.scannet_dir, 'pcd_with_object_aabbs', split, f'{scene_id}.pth')

            pcd_data = torch.load(pcd_path)
            pcd = pcd_data['xyz']

            pcd_discrete = video_processor.discrete_point(pcd)
            pcd_unique = list(set([tuple(x) for x in pcd_discrete]))
            scan2pcd[scene_id] = pcd_unique
    
    with open(os.path.join(args.output_dir, f"pcd_discrete_{args.voxel_size}.pkl"), "wb") as f:
        pickle.dump(scan2pcd, f) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_dir", type=str, default="data/scannet/")
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="data/metadata")
    args = parser.parse_args()

    main(args)