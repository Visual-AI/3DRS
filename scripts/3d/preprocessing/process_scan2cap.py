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

    for split in ["train", "val"]:
        print(f"Processing {split} split...")
        n_miss = 0
        with open(os.path.join(args.scanrefer_dir, f"ScanRefer_filtered_{split}.json")) as f:
            data = json.load(f)
        
        # preprocess annotations for val split. for each instance, there are many annotations
        if split == "val":
            instance_annotations = defaultdict(list)
            for item in data:
                key = f"{item['scene_id']}|{item['object_id']}|{item['object_name']}"
                instance_annotations[key].append(item['description'])

        scan2box = {}
        scan2pred = {}
        all_data = []
        visible_instances = set()
        for i, item in enumerate(tqdm(data)):
            scene_id = item['scene_id']

            # skip duplicate
            key = f"{item['scene_id']}|{item['object_id']}|{item['object_name']}"
            if split != 'train' and key in visible_instances:
                continue
            visible_instances.add(key)

            # load ground truth box
            if scene_id not in scan2box:
                scan2box[scene_id] = load_scene(os.path.join(args.scannet_dir, "pcd_with_object_aabbs", split, f"{item['scene_id']}.pth"))

            gt_box = scan2box[scene_id][item['object_id']]

            # load predicted boxes
            if split == "val":   
                if scene_id not in scan2pred:
                    pcd_data = torch.load(os.path.join(args.scannet_dir,
                                    'pcd_with_object_aabbs', split, f'{scene_id}.pth'))
                    points, colors = pcd_data["xyz"], pcd_data["rgb"]
                    colors = colors / 127.5 - 1
                    pcds = np.concatenate([points, colors], 1)    

                    pred_pcds = load_masks(args.scannet_dir, scene_id, pcds)
                    scan2pred[scene_id] = [convert_pc_to_box(pcd) for pcd in pred_pcds]

                boxes = scan2pred[scene_id]

                select_box = None
                max_iou = 0
                for center, sz in boxes:
                    iou = box3d_iou(
                        get_3d_box_corners(center, sz),
                        get_3d_box_corners(gt_box[:3], gt_box[3:])
                    )
                    if iou >= args.threshold:
                        if iou > max_iou:
                            max_iou = iou
                            select_box = center + sz
                
                if select_box is None:
                    print(f"{key} is missing")
                    n_miss += 1

            desc = item['description'].capitalize()
            new_item = {
                "id": i,
                "video": f"scannet/{item['scene_id']}",
                "conversations": [
                    {
                        "value": f"<image> Given an object located at <coord> , describe the object in detail.",
                        "from": "human",
                    },
                    {
                        "value": f"{desc}",
                        "from": "gpt",
                    },
                ],
                "box_input": gt_box if split == 'train' else select_box,
                "gt_box": gt_box,
                "metadata": {
                    "dataset": "scan2cap",
                    "question_type": item["eval_type"], 
                    "ann_id": item["ann_id"],
                    "object_id": item["object_id"],
                }
            }

            if split == "val":
                new_item['annotations'] = instance_annotations[key]

            all_data.append(new_item)
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'scan2cap_{split}_llava_style.json'), 'w') as f:
            json.dump(all_data, f)
        print(f"total {len(all_data)} items.")
        print(f"total {n_miss} miss.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanrefer_dir", type=str, default="data/benchmark/scanrefer/")
    parser.add_argument("--scannet_dir", type=str, default="data/scannet/")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args)