import re
import json
import argparse
import string
from tqdm import tqdm
import numpy as np
from llava.eval.box_utils import get_3d_box_corners, box3d_iou

def main(args):
    with open(args.input_file) as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    
    if args.n != -1:
        data = data[:args.n]

    iou_025_sum = 0
    iou_05_sum = 0

    from collections import defaultdict

    results_iou25 = defaultdict(list)
    results_iou50 = defaultdict(list)

    for item in tqdm(data):
        gt = item['gt_response']
        pred = item['pred_response']

        gt_corners = get_3d_box_corners(gt[:3], gt[3:])
        pred_corners = get_3d_box_corners(pred[:3], pred[3:])

        iou = box3d_iou(gt_corners, pred_corners)
        
        results_iou25["all"].append(iou >= 0.25)
        results_iou50["all"].append(iou >= 0.5)

        type1 = "easy" if "easy" in item['question_type'] else "hard"
        results_iou25[type1].append(iou >= 0.25)
        results_iou50[type1].append(iou >= 0.5)

        type2 = "view-dep" if "view-dep" in item['question_type'] else "vide-indep"
        results_iou25[type2].append(iou >= 0.25)
        results_iou50[type2].append(iou >= 0.5)


    for k in results_iou25.keys():
        print("===============================================")
        print(f'{k} IoU25: {np.mean(results_iou25[k])}')
        print(f'{k} IoU50: {np.mean(results_iou50[k])}')
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='results/nr3d/test.jsonl')
    parser.add_argument("-n", type=int, default=-1)
    args = parser.parse_args()

    main(args)