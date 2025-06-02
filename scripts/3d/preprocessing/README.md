# Data Preparation
We have provided the processed data in [Hugging Face](https://huggingface.co/datasets/zd11024/Video-3D-LLM_data).
The directory should be orgainized as 
```
Video-3D-LLM # project root
├── data
│   ├── scannet
│   │   ├── scans
│   │   ├── posed_images
│   │   ├── pcd_with_object_aabbs
│   │   └── mask
│   ├── embodiedscan
│   │   ├── embodiedscan_infos_train.pkl
│   │   ├── embodiedscan_infos_val.pkl
│   │   └── embodiedscan_infos_test.pkl
│   ├── metadata
│   │   ├── scannet_select_frames.json
│   │   ├── pcd_discrete_0.1.pkl
│   │   ├── scannet_train_gt_box.json
│   │   └── scannet_val_pred_box.json
│   ├── prcoessed
│   │   ├── multi3drefer_train_llava_style.json
│   │   ├── multi3drefer_val_llava_style.json
│   │   ├── ...
```
## Preprocessing
### ScanNet v2
1. Download the ScanNet v2 dataset [here](http://www.scan-net.org/). The folder of ScanNet should look like
```
Video-3D-LLM # project root
├── data
│   ├── scannet
│   │   ├── scans
│   │   │   ├── [scene_id]
│   │   │   │   ├── [scene_id]_vh_clean_2.ply
│   │   │   │   ├── [scene_id]_vh_clean_2.0.010000.segs.json
│   │   │   │   ├── [scene_id].aggregation.json
│   │   │   │   ├── [scene_id].txt
│   │   │   │   └── [scene_id].sens
```

2. Extract color images, depth images and camera parameter using the following script, which is modified from [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan/blob/main/embodiedscan/converter/generate_image_scannet.py).
```bash
python scripts/3d/preprocessing/generate_image_scannet.py --fast
```

3. Extract point clouds for each scene.
```bash
python scripts/3d/preprocessing/extract_scannet_pcd.py
```
This will generate the point clouds and object bounding boxes for each scan.


### EmbodiedScan
Download EmbodiedScan data at this [link](https://github.com/OpenRobotLab/EmbodiedScan/tree/main/data). You need to fill out the [official form](https://docs.google.com/forms/d/e/1FAIpQLScUXEDTksGiqHZp31j7Zp7zlCNV7p_08uViwP_Nbzfn3g6hhw/viewform) to get the access to the dataset. Decompress the embodiedscan and the directory should be orgainized as
```
├── data
│   ├── metadata
│   │   ├── embodiedscan
│   │   │   ├── embodiedscan_infos_train.pkl
│   │   │   ├── embodiedscan_infos_val.pkl
│   │   │   └── embodiedscan_infos_test.pkl
```

### Meta Information
1. Prepare the object proposals. For training set, we directly use the ground truth via the following command.
```bash
python scripts/3d/preprocessing/extract_gt_box.py
```
For the validation set, we utilize the object proposals detected by Mask3D. LEO provided the corresponding annotation results [here](https://huggingface.co/datasets/huangjy-pku/LEO_data/blob/main/mask.zip). We place it at `data/scannet/mask` and process it using the following script.
```bash
python scripts/3d/preprocessing/extract_pred_box.py
```

2. Prepare the maximum coverage sampling. Firstly we need to preprocess the voxel for each scan for maximum coverage sampling. The results will be saved at `data/metadata/pcd_discrete_0.1.pkl`.
```bash
python scripts/3d/preprocessing/convert_pcd_to_voxel.py
```
And then we perform the maximum coverage sampling offiline, and the results will be saved at `data/metadata/scannet_select_frames.json`.
```
python scripts/3d/preprocessing/max_coverage_sampling.py
```

### Downstream Benchmarks
1. SQA3D: Download the [SQA3D](https://github.com/SilongYong/SQA3D?tab=readme-ov-file) and convert the annotation to the LLaVA format using the following script.
```bash
python scripts/3d/preprocessing/process_sqa3d.py
```

2. ScanQA: Download the [ScanQA](https://github.com/ATR-DBI/ScanQA/blob/main/docs/dataset.md) and convert the annotation using the following script.
```bash
python scripts/3d/preprocessing/process_scanqa.py
```

3. ScanRefer: Download the [ScanRefer](https://daveredrum.github.io/ScanRefer/), and then run the following command.
```bash
python scripts/3d/preprocessing/process_scanrefer.py
```

4. Scan2Cap: Convert the annotation of ScanRefer to Scan2Cap.
```bash
python scripts/3d/preprocessing/process_scan2cap.py
```

5. Multi3DRefer: Download the [Multi3DRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP).
```bash
python scripts/3d/preprocessing/process_multi3drefer.py
```