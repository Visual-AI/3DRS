import os
import json
import torch
import argparse
from tqdm import tqdm


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
        template = "<image>Identify the object according to the following description.\n{desc}\nThere may be no corresponding object, or there may be one or more objects."

    for split in ["train", "val"]:
        with open(os.path.join(args.multi3drefer_dir, f"multi3drefer_{split}.json")) as f:
            data = json.load(f)
        
        all_data = []
        scan2box = {}
        for i, item in enumerate(tqdm(data)):
            if split == "test":
                box = None
            else:
                # load ground truth box
                scene_id = item['scene_id']
                if scene_id not in scan2box:
                    scan2box[scene_id] = load_scene(os.path.join('data/scannet', "pcd_with_object_aabbs", split, f"{scene_id}.pth"))
                # box = scan2box[scene_id][item['object_id']]
                box_list = [
                    scan2box[scene_id][str(object_id)]
                    for object_id in item['object_ids']
                ]
    
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
                "box": box_list,
                "metadata": {
                    "dataset": "multi3drefer",
                    "question_type": item["eval_type"], 
                    "ann_id": item["ann_id"],
                    "object_id": item["object_ids"],
                }
            })
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'multi3drefer_{split}_llava_style.json'), 'w') as f:
            json.dump(all_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi3drefer_dir", type=str, default="data/benchmark/multi3drefer/")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--template_type", type=str, default="cls")
    args = parser.parse_args()

    main(args)