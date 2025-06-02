sh scripts/3d/train/train_multi.sh
sh scripts/3d/eval/eval_scanrefer.sh llavanext-qwen-3drs uniform 32
sh scripts/3d/eval/eval_multi3drefer.sh llavanext-qwen-3drs uniform 32
sh scripts/3d/eval/eval_sqa3d.sh llavanext-qwen-3drs uniform 32
sh scripts/3d/eval/eval_scan2cap.sh llavanext-qwen-3drs uniform 32
sh scripts/3d/eval/eval_scanqa.sh llavanext-qwen-3drs uniform 32