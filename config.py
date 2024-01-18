import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='HardestContrastiveLossTrainer')
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
trainer_arg.add_argument('--batch_size', type=int, default=4)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

# Hard negative mining
trainer_arg.add_argument('--use_hard_negative', type=str2bool, default=True)
trainer_arg.add_argument('--hard_negative_sample_ratio', type=int, default=0.05)
trainer_arg.add_argument('--hard_negative_max_num', type=int, default=3000)
trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=256)

# Metric learning loss
trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--finest_thresh', type=float, default=0.2)
trainer_arg.add_argument('--pos_weight', type=float, default=1)
trainer_arg.add_argument('--neg_weight', type=float, default=1)
trainer_arg.add_argument('--finest_weight', type=float, default=1)
trainer_arg.add_argument('--block_finest_gradient', type=str2bool, default=True)
trainer_arg.add_argument('--use_group_circle_loss', type=str2bool, default=False)
trainer_arg.add_argument('--safe_radius', type=float, default=0.75)
trainer_arg.add_argument('--square_loss', type=str2bool, default=True)

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)
trainer_arg.add_argument('--max_in_p', type=int, default=20000)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

trainer_arg.add_argument('--stat_freq', type=int, default=40)
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)

trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)
trainer_arg.add_argument('--min_sample_frame_dist', type=float, default=10.0)
trainer_arg.add_argument('--complement_pair_dist', type=float, default=10.0)
trainer_arg.add_argument('--num_complement_one_side', type=int, default=5)

# Triplets
trainer_arg.add_argument('--triplet_num_pos', type=int, default=256)
trainer_arg.add_argument('--triplet_num_hn', type=int, default=512)
trainer_arg.add_argument('--triplet_num_rand', type=int, default=1024)

# dNetwork specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='ResUNetFatBN')
net_arg.add_argument('--encoder_model', type=str, default='ResUNetFatBN')
net_arg.add_argument('--model_n_out', type=int, default=32, help='Feature dimension')
net_arg.add_argument('--conv1_kernel_size', type=int, default=5)
net_arg.add_argument('--normalize_feature', type=str2bool, default=True)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='feat_match_ratio')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--loss_ratio', type=float, default=1e-5)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument(
    '--icp_cache_path', type=str, default="/home/chrischoy/datasets/FCGF/kitti/icp/")

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default=None)
misc_arg.add_argument('--train_num_thread', type=int, default=4)
misc_arg.add_argument('--val_num_thread', type=int, default=1)
misc_arg.add_argument('--test_num_thread', type=int, default=2)
misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument(
    '--nn_max_n',
    type=int,
    default=2000,
    help='The maximum number of features to find nearest neighbors in batch')

# Dataset specific configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ThreeDMatchPairDataset')
data_arg.add_argument('--train_dataset', type=str, default='ColocationKittiDataset')
data_arg.add_argument('--voxel_size', type=float, default=0.025)
data_arg.add_argument('--random_dist', type=str2bool, default=True)
data_arg.add_argument(
    '--threed_match_dir', type=str, default="/home/chrischoy/datasets/FCGF/threedmatch")
data_arg.add_argument(
    '--kitti_root', type=str, default="/home/chrischoy/datasets/FCGF/kitti/")
data_arg.add_argument(
    '--kitti_max_time_diff',
    type=int,
    default=3,
    help='max time difference between pairs (non inclusive)')
data_arg.add_argument('--kitti_date', type=str, default='2011_09_26')
data_arg.add_argument('--pair_min_dist', type=int, default=-1)
data_arg.add_argument('--pair_max_dist', type=int, default=-1)
data_arg.add_argument('--mutate_neighbour_percentage', type=float, default=0.)
data_arg.add_argument('--LoKITTI', type=str2bool, default=False)
data_arg.add_argument('--min_dist', type=int, default=5)
data_arg.add_argument('--max_dist', type=int, default=60)
data_arg.add_argument('--num_neighborhood', type=int, default=6)

# Debug configurations, placed in order to find the best hyperparameters
debug_arg = add_argument_group('Debug')
debug_arg.add_argument('--use_old_pose', type=str2bool, default=True)
debug_arg.add_argument('--debug_need_complement', type=str2bool, default=True)
debug_arg.add_argument('--debug_force_icp_recalculation', type=str2bool, default=False)
# debug_arg.add_argument('--debug_manual_seed', type=str2bool, default=False)
debug_arg.add_argument('--debug_use_old_complement', type=str2bool, default=False)
debug_arg.add_argument('--debug_downsample_ratio', type=float, default=1)
debug_arg.add_argument('--debug_floating_loss_ratio', type=str2bool, default=False)
debug_arg.add_argument('--debug_inverse_floating_loss_ratio', type=str2bool, default=False)
debug_arg.add_argument('--debug_matching_based_weighed_chamfer', type=str2bool, default=False)
debug_arg.add_argument('--finetune_restart', type=str2bool, default=False)
debug_arg.add_argument('--use_next_frame', type=str2bool, default=False)
debug_arg.add_argument('--calc_distance_err', type=str2bool, default=False)
debug_arg.add_argument('--use_pair_group_positive_loss', type=str2bool, default=False)


def get_config():
  args = parser.parse_args()
  return args
