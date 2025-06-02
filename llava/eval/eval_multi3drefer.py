import re
import json
import torch
import argparse
import string
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment
from llava.eval.box_utils import get_3d_box_corners, box3d_iou
from llava.video_utils import VideoProcessor


def evaluate_one_query(pred_info, gt_info):
    pred_bboxes_count = len(pred_info)
    gt_bboxes_count = len(gt_info)

    if pred_bboxes_count == 0 and gt_bboxes_count == 0:
        return 1, 1
    if pred_bboxes_count == 0 or gt_bboxes_count == 0:
        return 0, 0

    # initialize true positives
    iou25_tp = 0
    iou50_tp = 0

    # initialize the cost matrix
    square_matrix_len = max(gt_bboxes_count, pred_bboxes_count)
    iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float32)
    # TODO: convert to batch process
    for i, pred_box in enumerate(pred_info):
        for j, gt_box in enumerate(gt_info):
            obj_center, obj_box_size = pred_box[:3], pred_box[3:]
            gt_center, gt_box_size = gt_box[:3], gt_box[3:]
            iou_matrix[i, j] = box3d_iou(get_3d_box_corners(obj_center, obj_box_size),
                                        get_3d_box_corners(gt_center, gt_box_size))

    # apply matching algorithm
    row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)

    # iterate matched pairs, check ious
    for i in range(pred_bboxes_count):
        iou = iou_matrix[row_idx[i], col_idx[i]]
        # calculate true positives
        if iou >= 0.25:
            iou25_tp += 1
        if iou >= 0.5:
            iou50_tp += 1

    # calculate precision, recall and f1-score for the current scene
    iou25_f1_score = 2 * iou25_tp / (pred_bboxes_count + gt_bboxes_count)
    iou50_f1_score = 2 * iou50_tp / (pred_bboxes_count + gt_bboxes_count)
    return iou25_f1_score, iou50_f1_score



def main(args):

    with open(args.input_file) as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    
    if args.n != -1:
        data = data[:args.n]

    iou_025_sum = 0
    iou_05_sum = 0


    # video_processor = VideoProcessor(
    #     annotation_dir="/mnt/data0/research_data/zd/data/embodiedscan",
    #     voxel_size=0.1,
    #     min_xyz_range=[-15, -15, -5],
    #     max_xyz_range=[15, 15, 5]
    # )
    

    from collections import defaultdict
    iou25_f1_per_type = defaultdict(list)
    iou50_f1_per_type = defaultdict(list)

    predict_as_zt = 0
    predict_as_st = 0
    predict_as_mt = 0

    for item in tqdm(data):
        gt = item['gt_response']

        # scores = item["scores"]
        # objects = item["objects"]
        # pred = []        
        # if torch.max(torch.tensor(scores), dim=0)[1].item() != len(scores) - 1:
        #     scores = torch.sigmoid((torch.tensor(scores) - max(scores)) / 0.07)
        #     for box, score in zip(objects, scores[:-1]):
        #         if score >= args.threshold:
        #             pred.append(box)
        
        scores = torch.tensor(item["scores"])
        objects = item["objects"]
        pred = []
        if torch.max(scores, dim=0)[1].item() != len(scores) - 1:
            scores = torch.nn.functional.softmax(scores / 0.07)[:-1]
            sorted_scores, indices = torch.sort(scores, descending=True)
            cur_sum = 0
            for idx, s in zip(indices, sorted_scores):
                cur_sum += s
                pred.append(objects[idx])
                if cur_sum >= args.threshold:
                    break

        iou25_f1_score, iou50_f1_score = evaluate_one_query(pred, gt)
        
        iou25_f1_per_type["all"].append(iou25_f1_score)
        iou50_f1_per_type["all"].append(iou50_f1_score)
        iou25_f1_per_type[item['question_type']].append(iou25_f1_score)
        iou50_f1_per_type[item['question_type']].append(iou50_f1_score)

    #     if item["question_type"].startswith("st"):
    #         if len(pred) == 0:
    #             predict_as_zt += 1
    #         elif len(pred) > 1:
    #             predict_as_mt += 1
    #         else:
    #             predict_as_st += 1
    
    # print(predict_as_zt, predict_as_st, predict_as_mt)


    for k in iou25_f1_per_type.keys():
        print("===============================================")
        print(f'{k} IoU25: {np.mean(iou25_f1_per_type[k])}')
        print(f'{k} IoU50: {np.mean(iou50_f1_per_type[k])}')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='results/multi3drefer/val.jsonl')
    parser.add_argument("-n", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    main(args)