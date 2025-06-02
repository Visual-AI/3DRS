# modified from https://github.com/3dlg-hcvc/M3DRef-CLIP/blob/main/dataset/scannetv2/preprocess_all_data.py
import os
import csv
import json
import torch
import argparse
import numpy as np
import open3d as o3d
from os import cpu_count
from functools import partial
from tqdm.contrib.concurrent import process_map


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

def read_axis_align_matrix(file_path):
    axis_align_matrix = None
    with open(file_path, "r") as f:
        for line in f:
            line_content = line.strip()
            if 'axisAlignment' in line_content:
                axis_align_matrix = [float(x) for x in line_content.strip('axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    return axis_align_matrix


def read_mesh_file(file_path, axis_align_matrix):
    mesh = o3d.io.read_triangle_mesh(file_path)
    if axis_align_matrix is not None:
        mesh.transform(axis_align_matrix)  # align the mesh
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertices, dtype=np.float32), \
           np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8), \
           np.asarray(mesh.vertex_normals, dtype=np.float32)


def get_semantic_labels(obj_name_to_segs, seg_to_verts, num_verts, label_map, valid_semantic_mapping):
    # create a map, skip invalid labels to make the final semantic labels consecutive
    filtered_label_map = {}
    for i, valid_id in enumerate(valid_semantic_mapping):
        filtered_label_map[valid_id] = i

    semantic_labels = np.full(shape=num_verts, fill_value=-1, dtype=np.int8)  # max value: 127
    for label, segs in obj_name_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            if label_map[label] not in filtered_label_map:
                semantic_labels[verts] = 19
            elif label_map[label] == 22:
                semantic_labels[verts] = -1
            else:
                semantic_labels[verts] = filtered_label_map[label_map[label]]
    return semantic_labels


def read_agg_file(file_path, label_map, invalid_ids):
    object_id_to_segs = {}
    obj_name_to_segs = {}
    with open(file_path, "r") as f:
        data = json.load(f)
    for group in data['segGroups']:
        object_name = group['label']
        if object_name not in label_map:
            object_name = "case"  # TODO: randomly assign a name mapped to "objects"
        if label_map[object_name] in invalid_ids:
            # skip room architecture
            continue
        segments = group['segments']
        object_id_to_segs[group["objectId"]] = segments
        if object_name in obj_name_to_segs:
            obj_name_to_segs[object_name].extend(segments)
        else:
            obj_name_to_segs[object_name] = segments.copy()
    return object_id_to_segs, obj_name_to_segs


def read_seg_file(file_path):
    seg2verts = {}
    with open(file_path, "r") as f:
        data = json.load(f)
    for vert, seg in enumerate(data['segIndices']):
        if seg not in seg2verts:
            seg2verts[seg] = []
        seg2verts[seg].append(vert)
    return seg2verts


def get_instance_ids(object_id2segs, seg2verts, sem_labels):
    instance_ids = np.full(shape=len(sem_labels), fill_value=-1, dtype=np.int16)
    for object_id, segs in object_id2segs.items():
        for seg in segs:
            verts = seg2verts[seg]
            instance_ids[verts] = object_id
    return instance_ids


def get_aabbs(xyz, instance_ids):
    unique_inst_ids = np.unique(instance_ids)
    unique_inst_ids = unique_inst_ids[unique_inst_ids != -1]  # skip the invalid id
    num_of_aabb = unique_inst_ids.shape[0]
    aabb_corner_points = np.empty(shape=(num_of_aabb, 8, 3), dtype=np.float32)
    aabb_obj_ids = np.empty(shape=num_of_aabb, dtype=np.int16)

    combinations = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], copy=False), dtype=np.float32).T.reshape(-1, 3)
    for i, instance_id in enumerate(unique_inst_ids):
        point_indices = instance_ids == instance_id
        object_points = xyz[point_indices]
        min_corner = object_points.min(axis=0)
        max_corner = object_points.max(axis=0)
        corner_points = min_corner + (max_corner - min_corner) * combinations
        aabb_corner_points[i] = corner_points
        aabb_obj_ids[i] = instance_id
    return aabb_corner_points, aabb_obj_ids


def process_one_scene(scene, args, split, label_map, invalid_ids, valid_semantic_mapping):
    scan_path = os.path.join(args.scannet_dir, "scans") if split in ["train", "val"] else os.path.join(args.scannet_dir, "scans_test")
    mesh_file_path = os.path.join(scan_path, scene, scene + '_vh_clean_2.ply')
    agg_file_path = os.path.join(scan_path, scene, scene + '.aggregation.json')
    seg_file_path = os.path.join(scan_path, scene, scene + '_vh_clean_2.0.010000.segs.json')
    meta_file_path = os.path.join(scan_path, scene, scene + '.txt')

    # read meta_file
    axis_align_matrix = read_axis_align_matrix(meta_file_path)

    # read mesh file
    xyz, rgb, normal = read_mesh_file(mesh_file_path, axis_align_matrix)

    num_verts = len(xyz)

    sem_labels = None
    object_ids = None
    aabb_obj_ids = None
    aabb_corner_xyz = None
    if os.path.exists(agg_file_path) and os.path.exists(seg_file_path):
        # read seg_file
        seg2verts = read_seg_file(seg_file_path)
        # read agg_file
        object_id_to_segs, obj_name_to_segs = read_agg_file(agg_file_path, label_map, invalid_ids)
        # get semantic labels
        sem_labels = get_semantic_labels(obj_name_to_segs, seg2verts, num_verts, label_map, valid_semantic_mapping)
        # get instance ids
        object_ids = get_instance_ids(object_id_to_segs, seg2verts, sem_labels)
        # get axis-aligned bounding boxes
        aabb_corner_xyz, aabb_obj_ids = get_aabbs(xyz, object_ids)

    torch.save(
        {"xyz": xyz, "rgb": rgb, "normal": normal, "sem_labels": sem_labels,
         "instance_ids": object_ids, "aabb_obj_ids": aabb_obj_ids, "aabb_corner_xyz": aabb_corner_xyz},
        os.path.join(args.output_dir, split, f"{scene}.pth")
    )



def main(args):
    max_workers = min(cpu_count(), args.nproc)
    print(f"\nUsing {max_workers} CPU threads.")

    # read semantic label mapping file
    label_map = get_semantic_mapping_file(os.path.join(args.scannet_meta, "scannetv2-labels.combined.tsv"),
                                          args.semantic_mapping_name)

    for split in ("train", "val", "test"):
        output_path = os.path.join(args.output_dir, split)
        if os.path.exists(output_path):
            print(f"\nWarning: output directory {os.path.abspath(output_path)} exists, "
                  f"existing files will be overwritten.")
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(args.scannet_meta, f"scannetv2_{split}.txt")) as f:
            split_list = [line.strip() for line in f]
        print(f"==> Processing {split} split ...")
        process_map(
            partial(
                process_one_scene, args=args, split=split, label_map=label_map,
                invalid_ids=args.invalid_semantic_labels,
                valid_semantic_mapping=args.valid_semantic_mapping
            ), split_list, chunksize=1, max_workers=max_workers
        )
        print(f"==> Complete. Saved at: {os.path.abspath(output_path)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--scannet_dir", type=str, default="data/scannet")
    parser.add_argument("--output_dir", type=str, default="data/scannet/pcd_with_object_aabbs")
    parser.add_argument("--scannet_meta", type=str, default="scripts/3d/preprocessing/scannet_metadata")
    args = parser.parse_args()

    args.semantic_mapping_name = "nyu40"
    args.valid_semantic_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]  # skip floor, wall and ceiling
    args.invalid_semantic_labels = []

    main(args)