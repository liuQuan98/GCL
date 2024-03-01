#! /bin/bash
export PATH_POSTFIX=$1
export MISC_ARGS=$2

export CUDA_VISIBLE_DEVICES=0

export KITTI_PATH="/mnt/disk/NUSCENES/nusc_kitti"
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export VERSION=$(git rev-parse HEAD)
export OUT_DIR=${OUT_DIR:-./outputs/Experiments/PairComplementNuscenesDataset-v0.3/FinestContrastiveLossTrainer/ResUNetFatBN/SGD-lr1e-1-e200-b4i1-modelnout32/2022-12-29_15-53-27}
export PYTHONUNBUFFERED="True"

echo $OUT_DIR

mkdir -m 755 -p $OUT_DIR

LOG=${OUT_DIR}/log_${TIME}.txt

echo "Host: " $(hostname) | tee -a $LOG
echo "Conda " $(which conda) | tee -a $LOG
echo $(pwd) | tee -a $LOG
echo "Version: " $VERSION | tee -a $LOG
# echo "Git diff" | tee -a $LOG
# echo "" | tee -a $LOG
# git diff | tee -a $LOG
echo "" | tee -a $LOG
nvidia-smi | tee -a $LOG


# Test
python -m scripts.test_kitti \
	--kitti_root ${KITTI_PATH} \
	--LoNUSCENES false \
	--rre_thresh 5 \
	--rte_thresh 2 \
	--pair_min_dist 10 \
	--pair_max_dist 20 \
	--downsample_single 1.0 \
	--use_RANSAC true \
	--save_dir ${OUT_DIR} | tee -a $LOG
