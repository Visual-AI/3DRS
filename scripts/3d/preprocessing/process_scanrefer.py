import os
import csv
import json
import torch
import argparse
from tqdm import tqdm

# modified from https://github.com/3dlg-hcvc/M3DRef-CLIP/blob/main/dataset/scanrefer/add_evaluation_labels.py
def get_semantic_mapping_file(file_path, mapping_name):
    label_mapping = {}
    mapping_col_idx = {
        "nyu40": 4,
        "eigen13": 5,
        "mpcat40": 16
    }
    with open(file_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        next(tsv_file)  # skip the header
        for line in tsv_file:
            label_mapping[line[1]] = int(line[mapping_col_idx[mapping_name]])
    return label_mapping


def add_unique_multiple_labels_to_json(file_path, label_mapping, valid_semantic_mapping):
    with open(file_path, "r") as f:
        scanrefer_json_data = json.load(f)
    obj_cache = {}
    sem_cache = {}
    for item in scanrefer_json_data:
        if (item["scene_id"], item["object_id"]) in obj_cache:
            continue
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39
        if (item['scene_id'], sem_label) not in sem_cache:
            sem_cache[(item['scene_id'], sem_label)] = 0
        sem_cache[(item['scene_id'], sem_label)] += 1
        obj_cache[(item["scene_id"], item["object_id"])] = True

    for item in scanrefer_json_data:
        scene_id = item['scene_id']
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39
        assert sem_cache[(scene_id, sem_label)] >= 1
        item["eval_type"] = "unique" if sem_cache[(scene_id, sem_label)] == 1 else "multiple"
    # save the new json
    with open(file_path, "w") as f:
        json.dump(scanrefer_json_data, f, indent=2)


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

        ret[object_id] = (x_center, y_center, z_center, w, h, l)

    return ret


def main(args):


    if args.template_type == "gen":
        template = "<image>Localize the object according to the following description.\n{desc}\nOutput the answer in the format x_min,y_min,z_min,x_max,y_max,z_max",
    elif args.template_type == "cls":
        template = "<image>Identify the object according to the following description.\n{desc}"

    for split in ["train", "val"]:

        with open(os.path.join(args.scanrefer_dir, f"ScanRefer_filtered_{split}.json")) as f:
            data = json.load(f)
        
        all_data = []
        scan2box = {}
        for i, item in enumerate(tqdm(data)):
            scan2obj = {}
            if split == "test":
                box = None
            else:
                # load ground truth box
                scene_id = item['scene_id']
                if scene_id not in scan2box:
                    scan2box[scene_id] = load_scene(os.path.join('data/scannet', "pcd_with_object_aabbs", split, f"{scene_id}.pth"))
                box = scan2box[scene_id][item['object_id']]
            desc = item['description'].capitalize()
            all_data.append({
                "id": i,
                "video": f"scannet/{item['scene_id']}",
                "conversations": [
                    {
                        "value": template.format(desc=desc),
                        "from": "human",
                    },
                    {
                        "value": f"<ground>",
                        "from": "gpt",
                    },
                ],
                "box": box,
                "metadata": {
                    "dataset": "scanrefer",
                    "question_type": item["eval_type"], 
                    "ann_id": item["ann_id"],
                    "object_id": item["object_id"],
                }
            })
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'scanrefer_vg_{split}_llava_style.json'), 'w') as f:
            json.dump(all_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanrefer_dir", type=str, default="data/benchmark/scanrefer/")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--template_type", type=str, default="cls")
    args = parser.parse_args()

    args.label_mapping_file = "scripts/3d/preprocessing/scannet_metadata/scannetv2-labels.combined.tsv"
    args.valid_semantic_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]  # skip floor, wall and ceiling

    for split in ["train", "val"]:
        label_mapping = get_semantic_mapping_file(args.label_mapping_file, "nyu40")
        add_unique_multiple_labels_to_json(
            os.path.join(args.scanrefer_dir, f"ScanRefer_filtered_{split}.json"),
            label_mapping,
            args.valid_semantic_mapping,
        )

    main(args)