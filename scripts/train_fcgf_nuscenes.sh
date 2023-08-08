#! /bin/bash
export PATH_POSTFIX=$1
export MISC_ARGS=$2

export CUDA_VISIBLE_DEVICES=2

export KITTI_PATH="/mnt/disk/NUSCENES/nusc_kitti"

export DATA_ROOT="./outputs/Experiments"
export DATASET=${DATASET:-PairComplementNuscenesDataset}
export TRAINER=${TRAINER:-HardestContrastiveLossTrainer}
export ENCODER_MODEL=${MODEL:-ResUNetBN2C}
export MODEL_N_OUT=${MODEL_N_OUT:-128}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-1}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export MAX_EPOCH=${MAX_EPOCH:-200}
export BATCH_SIZE=${BATCH_SIZE:-4}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export ITER_SIZE=${ITER_SIZE:-1}
export BEST_VAL_METRIC=${BEST_VAL_METRIC:-feat_match_ratio}
export VOXEL_SIZE=${VOXEL_SIZE:-0.3}
export POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER=${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER:-1.5}
export CONV1_KERNEL_SIZE=${CONV1_KERNEL_SIZE:-5}
export EXP_GAMMA=${EXP_GAMMA:-0.99}
export RANDOM_SCALE=${RANDOM_SCALE:-True}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export VERSION=$(git rev-parse HEAD)

export OUT_DIR=${DATA_ROOT}/${DATASET}-v${VOXEL_SIZE}/${TRAINER}/${MODEL}/${OPTIMIZER}-lr${LR}-e${MAX_EPOCH}-b${BATCH_SIZE}i${ITER_SIZE}-modelnout${MODEL_N_OUT}${PATH_POSTFIX}/${TIME}

# export OUT_DIR=${OUT_DIR:-./outputs/Experiments/PairComplementNuscenesDataset-v0.3/HardestContrastiveLossTrainer//SGD-lr1e-1-e200-b4i1-modelnout128/2022-02-15_19-52-13}
export PYTHONUNBUFFERED="True"

# export RESUME_FILE="./outputs/Experiments/PairComplementNuscenesDataset-v0.3/HardestContrastiveLossTrainer//SGD-lr1e-1-e200-b4i1-modelnout128/2022-01-28_13-39-04/checkpoint.pth"
# export RESUME_DIR="./outputs/Experiments/PairComplementNuscenesDataset-v0.3/HardestContrastiveLossTrainer//SGD-lr1e-1-e200-b4i1-modelnout128/2022-01-28_13-39-04"

echo $OUT_DIR
echo "Using GPU No:"
echo $CUDA_VISIBLE_DEVICES

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

# Training With Resume
python train.py \
	--dataset ${DATASET} \
	--train_dataset none \
	--trainer ${TRAINER} \
	--model ${ENCODER_MODEL} \
	--model_n_out ${MODEL_N_OUT} \
	--conv1_kernel_size ${CONV1_KERNEL_SIZE} \
	--optimizer ${OPTIMIZER} \
	--lr ${LR} \
	--batch_size ${BATCH_SIZE} \
	--val_batch_size ${VAL_BATCH_SIZE} \
	--iter_size ${ITER_SIZE} \
	--max_epoch ${MAX_EPOCH} \
	--voxel_size ${VOXEL_SIZE} \
	--out_dir ${OUT_DIR} \
	--use_random_scale ${RANDOM_SCALE} \
	--weight_decay ${WEIGHT_DECAY} \
	--positive_pair_search_voxel_size_multiplier ${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER} \
	--kitti_root ${KITTI_PATH} \
	--hit_ratio_thresh 0.3 \
	--exp_gamma ${EXP_GAMMA} \
	--complement_pair_dist 10 \
	--num_complement_one_side 4 \
	--best_val_metric ${BEST_VAL_METRIC} \
	--random_dist true \
	--debug_need_complement false \
	--debug_use_old_complement false \
	--use_old_pose true \
	--pair_min_dist 50 \
	--pair_max_dist 50 \
	# --resume ${RESUME_FILE} \
	# --resume_dir ${RESUME_DIR} \
	# --finetune_restart true \
	$MISC_ARGS 2>&1 | tee -a $LOG

