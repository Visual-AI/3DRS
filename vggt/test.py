import torch
from mast3r.model import AsymmetricMASt3R
from math import inf
from dust3r.utils.image import load_images
import os
import random
from tqdm import tqdm

model_str = "AsymmetricMASt3R(wpose=False, pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))"
model = eval(model_str)

weights_path = "checkpoints/geometry_pose.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(weights_path, map_location=device)
msg = model.load_state_dict(ckpt['model'], strict=False)
print('modeling loading status', msg)
model = model.to(device).eval()

root_dir = '/root/paddlejob/workspace/env_run/output/Video-3D-LLM/data/scannet/posed_images'
root_save_3d_feature = '/root/paddlejob/workspace/env_run/output/Video-3D-LLM/data/scannet/posed_images_3d_feature'
if not os.path.exists(root_save_3d_feature):
    os.makedirs(root_save_3d_feature)
    print("create dir", root_save_3d_feature)
scene_list = os.listdir(root_dir)
num_frames_to_sample = 32
for scene in tqdm(scene_list):
    if not os.path.isdir(os.path.join(root_save_3d_feature, scene)):
        os.makedirs(os.path.join(root_save_3d_feature, scene))
    
    file_dir = os.listdir(os.path.join(root_dir, scene))
    file_list = [os.path.join(root_dir, scene, file) for file in file_dir if file.endswith('.jpg')]
    file_list.sort()

    total_frames = len(file_list)
    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

    file_list = [file_list[i] for i in sampled_indices]

    batch = load_images(file_list, size=512, verbose=False)
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'rng', 'vid'])
    ignore_dtype_keys = set(['true_shape', 'camera_pose', 'pts3d', 'fxfycxcy', 'img_org', 'camera_intrinsics', 'depthmap', 'depth_anything', 'fxfycxcy_unorm'])
    dtype = torch.bfloat16
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], torch.Tensor):
                view[name] = view[name].to(device, non_blocking=True)
            else:
                view[name] = torch.tensor(view[name]).to(device, non_blocking=True)
            if view[name].dtype == torch.float32 and name not in ignore_dtype_keys:
                view[name] = view[name].to(dtype)

    view1 = batch[:1]
    view2 = batch[1:]
    # print(view1[0].keys(), view1[0]['img'].shape, view1[0]['true_shape'], view1[0]['idx'], view1[0]['instance'], view1[0]['img'])
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        outs = model(view1, view2, True, dtype)
        f = outs[-1]
        f = f.cpu().numpy()
        np.savez_compressed(os.path.join(root_save_3d_feature, scene, 'flare' + '.npz'), f)
