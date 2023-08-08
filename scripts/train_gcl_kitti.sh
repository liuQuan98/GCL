#! /bin/bash
export PATH_POSTFIX=$1
export MISC_ARGS=$2

export CUDA_VISIBLE_DEVICES=1

export KITTI_PATH="/mnt/disk/KITTIOdometry_Full"

export DATA_ROOT="./outputs/Experiments"
export DATASET=${DATASET:-PairComplementKittiDataset}
export TRAIN_DATASET=${TRAIN_DATASET:-ColocationKittiDataset}
export TRAINER=${TRAINER:-FinestContrastiveLossTrainer}
export MODEL=${MODEL:-ResUNetFatBN}
export MODEL_N_OUT=${MODEL_N_OUT:-32}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-1}
export LOSS_RATIO=${LOSS_RATIO:-2e-3}
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
# export KITTI_PATH=${KITTI_PATH:-/home/chrischoy/datasets/KITTI_FCGF}
export VERSION=$(git rev-parse HEAD)

export OUT_DIR=${DATA_ROOT}/${DATASET}-v${VOXEL_SIZE}/${TRAINER}/${MODEL}/${OPTIMIZER}-lr${LR}-e${MAX_EPOCH}-b${BATCH_SIZE}i${ITER_SIZE}-modelnout${MODEL_N_OUT}${PATH_POSTFIX}/${TIME}

# export OUT_DIR=${OUT_DIR:-./outputs/Experiments/PairComplementKittiDataset-v0.3/FinestContrastiveLossTrainer//SGD-lr1e-1-e200-b4i1-modelnout128/2022-10-25_11-01-02}

export PYTHONUNBUFFERED="True"

# export RESUME_FILE="outputs/Experiments/FCGF_pretrained/2023-02-13_19-48-37_FCGF_5_20/best_val_checkpoint.pth"
# export RESUME_DIR="outputs/Experiments/FCGF_pretrained/2023-02-13_19-48-37_FCGF_5_20"

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
	--train_dataset ${TRAIN_DATASET} \
	--trainer ${TRAINER} \
	--model ${MODEL} \
	--model_n_out ${MODEL_N_OUT} \
	--conv1_kernel_size ${CONV1_KERNEL_SIZE} \
	--optimizer ${OPTIMIZER} \
	--lr ${LR} \
	--loss_ratio ${LOSS_RATIO} \
	--batch_size ${BATCH_SIZE} \
	--val_batch_size ${VAL_BATCH_SIZE} \
	--iter_size ${ITER_SIZE} \
	--max_epoch ${MAX_EPOCH} \
	--voxel_size ${VOXEL_SIZE} \
	--out_dir ${OUT_DIR} \
	--use_random_scale ${RANDOM_SCALE} \
	--use_random_rotation true \
	--positive_pair_search_voxel_size_multiplier ${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER} \
	--weight_decay ${WEIGHT_DECAY} \
	--kitti_root ${KITTI_PATH} \
	--hit_ratio_thresh 0.3 \
	--exp_gamma ${EXP_GAMMA} \
	--complement_pair_dist 10 \
	--num_complement_one_side 3 \
	--best_val_metric ${BEST_VAL_METRIC} \
	--random_dist true \
	--debug_need_complement true \
	--debug_use_old_complement false \
	--use_old_pose false \
	--pair_min_dist 5 \
	--pair_max_dist 40 \
	--min_dist 5 \
	--max_dist 60 \
	--num_neighborhood 6 \
	--block_finest_gradient false \
	--finest_weight 1 \
	--pos_weight 1 \
	--calc_distance_err false \
	--use_hard_negative true \
	--use_pair_group_positive_loss false \
	--use_group_circle_loss false \
	--safe_radius 0.75 \
	--num_pos_per_batch 256 \
	--square_loss true \
	# --resume ${RESUME_FILE} \
	# --resume_dir ${RESUME_DIR} \
	# --finetune_restart true \
 	# --mutate_neighbour_percentage 0.0 \
	$MISC_ARGS 2>&1 | tee -a $LOG

