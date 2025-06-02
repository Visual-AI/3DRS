import re
import json
from collections import defaultdict
import numpy as np
import argparse
import string
from tqdm import tqdm
from llava.eval.box_utils import get_3d_box_corners, box3d_iou
from llava.video_utils import VideoProcessor, merge_video_dict

def main(args):
    with open(args.input_file) as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    
    if args.n != -1:
        data = data[:args.n]
    
    iou25_acc_per_type = defaultdict(list)
    iou50_acc_per_type = defaultdict(list)

    for item in tqdm(data):
        gt = item['gt_response']
        pred = item['pred_response']

        gt_corners = get_3d_box_corners(gt[:3], gt[3:])
        pred_corners = get_3d_box_corners(pred[:3], pred[3:])

        iou = box3d_iou(gt_corners, pred_corners)

        iou25_acc_per_type["all"].append(iou >= 0.25)
        iou50_acc_per_type["all"].append(iou >= 0.5)
        iou25_acc_per_type[item["question_type"]].append(iou >= 0.25)
        iou50_acc_per_type[item["question_type"]].append(iou >= 0.5)
        
    
    for k in iou25_acc_per_type:
        print(f"{k} iou@0.25: {np.mean(iou25_acc_per_type[k]) * 100}")
        print(f"{k} iou@0.5: {np.mean(iou50_acc_per_type[k]) * 100 }")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='results/scanrefer/val.jsonl')
    parser.add_argument("-n", type=int, default=-1)
    args = parser.parse_args()

    main(args)