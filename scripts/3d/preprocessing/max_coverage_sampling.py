import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from llava.video_utils import VideoProcessor
import torch
import random
import pickle

@torch.no_grad()
def main(args):

    with open(f'data/metadata/pcd_discrete_{args.voxel_size}.pkl', 'rb') as f:
        pc_data = pickle.load(f)

    video_processor = VideoProcessor(
        voxel_size=args.voxel_size,
        min_xyz_range=None,
        max_xyz_range=None,
    )

    all_data = []
    for i, scene_id in tqdm(enumerate(os.listdir(args.video_dir))):
        video_id = f"scannet/{scene_id}"
        meta_info = video_processor.scene[video_id]
        # frame_files = [os.path.join(video_processor.video_folder, img["img_path"]) for img in meta_info["images"]][::3]
        # if len(frame_files) < 32:
        frame_files = [os.path.join(video_processor.video_folder, img["img_path"]) for img in meta_info["images"]][::2]
        if len(frame_files) < 32:
            frame_files = [os.path.join(video_processor.video_folder, img["img_path"]) for img in meta_info["images"]]
        # print(len(frame_files))
        world_coords = video_processor.calculate_world_coords(
            video_id, 
            frame_files,
        )["world_coords"]

        world_coords_discrete = {}
        world_coords = world_coords.to('cuda')
        # world_coords = torch.maximum(world_coords, video_processor.min_xyz_range.to(world_coords.device))
        # world_coords = torch.minimum(world_coords, video_processor.max_xyz_range.to(world_coords.device))
        # world_coords = (world_coords - video_processor.min_xyz_range.to(world_coords.device)) 
        world_coords = world_coords / video_processor.voxel_size
        world_coords = world_coords.round().int()

        for frame_file, coords in zip(frame_files, world_coords):
            coords = coords.view(-1, 3)
            coords = torch.unique(coords, dim=0)
            world_coords_discrete[frame_file] = [tuple(c) for c in coords.tolist()]

        all_voxel = set()
        zs = [c[2] for v in world_coords_discrete.values() for c in v]
        zs = sorted(zs)
        zs = np.unique(zs)
        maxz = np.percentile(zs, 95)
        center_discrete = video_processor.discrete_point([0,0,0])

        for frame_file in world_coords_discrete:
            # world_coords_discrete[frame_file] = [
            #     c for c in world_coords_discrete[frame_file] if c[2] >= center_discrete[2] and c[2] <= maxz
            # ]
            all_voxel.update(world_coords_discrete[frame_file])

        pc_voxel = set(pc_data[scene_id])


        select_frame_files = []
        used_voxel = set()
        voxel_nums = []
        for _ in range(len(frame_files)):
            maxv = -1
            select_frame = []
            for frame_file in world_coords_discrete:
                cur_set = set(world_coords_discrete[frame_file]) & pc_voxel
                n_inter = len(used_voxel & cur_set)
                n_occupy = len(cur_set) - n_inter
                if n_occupy > maxv:
                    maxv = n_occupy
                    select_frame = [frame_file]
                elif n_occupy == maxv:
                    select_frame.append(frame_file)
            
            select_frame = random.choice(select_frame)

            used_voxel.update(world_coords_discrete[select_frame])
            select_frame_files.append(select_frame)
            voxel_nums.append(maxv)
            world_coords_discrete.pop(select_frame)

            # if len(used_voxel) / len(all_voxel) >=  0.95:
            #     break
            if len(select_frame_files) >= 32:
                break


        print(f"[{scene_id}] all voxels: {len(used_voxel & pc_voxel)}/{len(all_voxel & pc_voxel)} | ratio: {len(used_voxel & pc_voxel)/len(all_voxel & pc_voxel)} | image num: {len(select_frame_files)}/{len(frame_files)} | max: {maxz}")
        all_data.append({
            "video_id": video_id,
            "frame_files": select_frame_files,
            "voxel_nums": voxel_nums,
            "num_all_voxels": len(all_voxel & pc_voxel),
            "num_select_voxels": len(used_voxel & pc_voxel),
        })


    with open(args.output_file, "w") as f:
        json.dump(all_data, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--video_dir", type=str, default="data/scannet/posed_images")
    parser.add_argument("--output_file", type=str, default="data/metadata/scannet_select_frames.json")
    args = parser.parse_args()
    args.min_xyz_range = [-15, -15, -5]
    args.max_xyz_range = [15, 15, 5]

    main(args)
