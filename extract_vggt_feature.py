import os
import torch
import numpy as np
from tqdm import tqdm

# -------------------------------
# Import new model and related utility functions
# -------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# -------------------------------
# Device and data type settings
# -------------------------------
# If you have only one GPU, change "cuda:1" to "cuda:0"
device = "cuda:1" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    # Get the current GPU's compute capability: Ampere GPU (Compute Capability 8.0+) supports bfloat16
    dev_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    dtype = torch.bfloat16 if dev_capability[0] >= 8 else torch.float16
else:
    dtype = torch.float32

# -------------------------------
# Initialize the VGGT model and load pretrained weights
# -------------------------------
model = VGGT()
checkpoint = torch.load('checkpoints/model.pt', map_location=device)
msg = model.load_state_dict(checkpoint)
print('loading status:', msg)
model = model.to(device).eval()

# -------------------------------
# Data storage path settings
# -------------------------------
root_dir = 'data/scannet/posed_images'
root_save_3d_feature = 'data/scannet/posed_images_3d_feature_vggt'
if not os.path.exists(root_save_3d_feature):
    os.makedirs(root_save_3d_feature)
    print("create dir", root_save_3d_feature)

scene_list = os.listdir(root_dir)
num_frames_to_sample = 32

# -------------------------------
# Iterate over each scene and sample a fixed number of frames
# -------------------------------
for scene in tqdm(scene_list):
    scene_dir = os.path.join(root_dir, scene)
    if not os.path.isdir(scene_dir):
        continue

    # Ensure that each scene has a corresponding save directory
    scene_save_dir = os.path.join(root_save_3d_feature, scene)
    if not os.path.exists(scene_save_dir):
        os.makedirs(scene_save_dir)

    # Get all jpg images in the scene, sort them by filename, then sample a fixed number of frames
    file_names = [file for file in os.listdir(scene_dir) if file.endswith('.jpg')]
    file_names.sort()
    total_frames = len(file_names)
    if total_frames == 0:
        continue
    sampled_indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)
    sampled_file_list = [os.path.join(scene_dir, file_names[i]) for i in sampled_indices]

    # -------------------------------
    # Load images
    # -------------------------------
    # Here we use the load_and_preprocess_images function provided by vggt
    # The returned result is assumed to have shape (N, 3, H, W)
    images = load_and_preprocess_images(sampled_file_list)
    images = images.to(device, non_blocking=True)
    images = images.to(dtype)
    # Add batch dimension, final shape is (1, num_frames, 3, H, W)
    images = images.unsqueeze(0)

    # -------------------------------
    # Model inference, feature extraction
    # -------------------------------
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)

    # -------------------------------
    # Save aggregated_tokens_list[-1] and ps_idx
    # -------------------------------
    # Move the output to CPU and convert to numpy format
    feature = aggregated_tokens_list[-1].cpu().numpy()
    # Assume ps_idx is a tensor, otherwise save directly
    ps_idx_np = ps_idx.cpu().numpy() if isinstance(ps_idx, torch.Tensor) else ps_idx

    save_path = os.path.join(scene_save_dir, 'vggt.npz')
    np.savez_compressed(save_path, feature=feature, ps_idx=ps_idx_np)