#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --mem 50G
#SBATCH -c 10
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --error=errors_NC.o%j
#SBATCH --output=output_NC.o%j

echo "running in shell: " "$SHELL"
echo "*** loading spack modules ***"

#source ~/.bashrc
#conda activate optivist
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/p/ppowell/miniconda3/envs/optivist/lib/
#echo $LD_LIBRARY_PATH

#echo "*** set workdir ***"

/share/neurobiopsychologie/scratch/minconda3/envs/YOLO_v5_env/bin/python ultralytics/train.py --data ultralytics/data/subcoco_eghand.yaml --weights ultralytics/yolov5s.pt --img 640 --epochs 100 "$@"

