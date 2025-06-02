
"""
Refer to https://github.com/embodied-generalist/embodied-generalist/blob/main/data/datasets.py

scanrefer
- ScanRefer_filtered_train.json
- ScanRefer_filtered_val.json

scannet
- mask
- pcd_with_global_alignment
- vg

"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy import sparse
from collections import defaultdict
from llava.utils_3d import convert_pc_to_box
from llava.eval.box_utils import get_3d_box_corners, box3d_iou


def load_masks(scannet_base, scan_id, pcds):
    mask_path = os.path.join(scannet_base, 'mask', f'{str(scan_id)}.mask.npz')
    obj_masks = np.array(sparse.load_npz(mask_path).todense())[:50, :]
    obj_pcds_pred = []
    for i in range(obj_masks.shape[0]):
        mask = obj_masks[i]
        obj_pcds_pred.append(pcds[mask == 1, :].astype(float))

    return obj_pcds_pred


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

    for split in ["val"]:
      
        # load predicted boxes
        for split in ["val", "test"]:
            with open(os.path.join(args.scannet_meta, f"scannetv2_{split}.txt")) as f:
                scene_ids = [line.strip() for line in f]
            scan2obj = {}
            for scene_id in tqdm(scene_ids):
                try:
                    pcd_data = torch.load(os.path.join(args.scannet_dir,
                                    'pcd_with_object_aabbs', split, f'{scene_id}.pth'))
                    # points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
                    points, colors = pcd_data['xyz'], pcd_data['rgb']
                    colors = colors / 127.5 - 1
                    pcds = np.concatenate([points, colors], 1)    

                    pred_pcds = load_masks(args.scannet_dir, scene_id, pcds)
                    # scan2obj[scene_id] = [convert_pc_to_box(pcd) for pcd in pred_pcds]
                    scan2obj[f"scannet/{scene_id}"] = []
                    for pcd in pred_pcds:
                        center, sz = convert_pc_to_box(pcd)
                        scan2obj[f"scannet/{scene_id}"].append(center + sz)

                except Exception as e:
                    print(scene_id)
                    print(e)                


            os.makedirs(args.output_dir, exist_ok=True)
            filename = os.path.join(args.output_dir, f"scannet_{split}_pred_box.json")

            with open(filename, "w") as f:
                json.dump(scan2obj, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_dir", type=str, default="data/scannet/")
    parser.add_argument("--output_dir", type=str, default="data/metadata")
    parser.add_argument("--scannet_meta", type=str, default="scripts/3d/preprocessing/scannet_metadata")
    args = parser.parse_args()

    main(args)