# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash, _exhaustive_hash, _neg_hash, square_distance

import MinkowskiEngine as ME


class AlignmentTrainer:

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)

    if config.weights:
      checkpoint = torch.load(config.weights)
      model.load_state_dict(checkpoint['state_dict'])

    logging.info(model)

    self.config = config
    self.model = model
    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.optimizer = getattr(optim, config.optimizer)(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    self.test_valid = True if self.val_data_loader is not None else False
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        if not config.finetune_restart:
          self.start_epoch = state['epoch']
          self.scheduler.load_state_dict(state['scheduler'])
          self.optimizer.load_state_dict(state['optimizer'])
          if 'best_val' in state.keys():
            self.best_val = state['best_val']
            self.best_val_epoch = state['best_val_epoch']
            self.best_val_metric = state['best_val_metric']
        else:
          logging.info("=> Finetuning, will only load model weights.")
        model.load_state_dict(state['state_dict'])
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    # if self.test_valid:
    #   with torch.no_grad():
    #     val_dict = self._valid_epoch()

    #   for k, v in val_dict.items():
    #     self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if self.best_val < val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')
        elif self.best_val == val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the latest best val model (not overriding the first) with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self._save_checkpoint(epoch, 'best_val_newest_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

  def _save_checkpoint(self, epoch, filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)


class ContrastiveLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'].to(self.device),
            coordinates=input_dict['sinput0_C'].to(self.device))
        F0 = self.model(sinput0).F

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'].to(self.device),
            coordinates=input_dict['sinput1_C'].to(self.device))
        F1 = self.model(sinput1).F

        N0, N1 = len(sinput0), len(sinput1)

        pos_pairs = input_dict['correspondences']
        neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)
        pos_pairs = pos_pairs.long().to(self.device)
        neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)

        neg0 = F0.index_select(0, neg_pairs[:, 0])
        neg1 = F1.index_select(0, neg_pairs[:, 1])
        pos0 = F0.index_select(0, pos_pairs[:, 0])
        pos1 = F1.index_select(0, pos_pairs[:, 1])

        # Positive loss
        pos_loss = (pos0 - pos1).pow(2).sum(1)

        # Negative loss
        neg_loss = F.relu(self.neg_thresh -
                          ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size

        # Weighted loss
        loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the begining of iter_size
        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

      self.optimizer.step()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'].to(self.device),
          coordinates=input_dict['sinput0_C'].to(self.device))
      F0 = self.model(sinput0).F

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'].to(self.device),
          coordinates=input_dict['sinput1_C'].to(self.device))
      F1 = self.model(sinput1).F
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'][0], input_dict['pcd1'][0], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()


class FinestContrastiveLossTrainer(ContrastiveLossTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    ContrastiveLossTrainer.__init__(self, config, data_loader, val_data_loader)
    self.pos_weight = config.pos_weight
    self.finest_thresh = config.finest_thresh
    self.finest_weight = config.finest_weight
    self.block_finest_gradient = config.block_finest_gradient
    self.calc_distance_err = config.calc_distance_err
    self.use_hard_negative = config.use_hard_negative
    self.use_pair_group_positive_loss = config.use_pair_group_positive_loss
    self.log_scale = 16
    self.safe_radius = config.safe_radius
    self.square_loss = config.square_loss

    if config.use_group_circle_loss:
      self.location_loss = self.location_circle_loss
    elif self.finest_weight != 0:
      self.location_loss = self.finest_contrastive_loss
    else:
      self.location_loss = self.location_contrastive_loss

  def finest_contrastive_loss(self,
                              F_out,
                              group,
                              index,
                              index_hash,
                              finest_flag,
                              max_pos_cluster=256,
                              max_hn_samples=2048,
                              points=None,
                              batch_lengths=None):
    """
    Calculates the finest, positive, and negative losses of input co-location groups.
    Works only with proper data input from colocation_data_loader.
    """
    group, index, finest_flag = group.to(self.device), index.to(self.device), finest_flag.to(self.device)
    N_out = len(F_out)
    hash_seed = N_out
    split_timer, pos_timer, dist_timer, hash2_timer, neg_timer = Timer(), Timer(), Timer(), Timer(), Timer()

    # positive loss and finest loss
    split_timer.tic()
    pos_loss = 0
    finest_loss = 0
    N_pos_clusters = len(group)
    index_split = torch.split(index, tuple(group.tolist()))
    finest_flag_split = torch.split(finest_flag, tuple(group.tolist()))
    if N_pos_clusters > max_pos_cluster:
      pos_sel = np.random.choice(N_pos_clusters, max_pos_cluster, replace=False)
    else:
      pos_sel = np.arange(N_pos_clusters)
    split_timer.toc()

    pos_timer.tic()
    for i in pos_sel:
      index_set, finest_flag_set = index_split[i], finest_flag_split[i]
      feature_set = F_out[index_set]
      if self.use_pair_group_positive_loss:
        pos_positions = np.random.choice(len(feature_set), 2, replace=False)
        if self.square_loss:
          pos_loss += F.relu((feature_set[pos_positions[0]] - feature_set[pos_positions[1]]).pow(2).sum(-1) - self.pos_thresh)
        else:
          pos_loss += F.relu(torch.sqrt((feature_set[pos_positions[0]] - feature_set[pos_positions[1]]).pow(2).sum(-1) + 1e-7) - self.pos_thresh)
      else:
        if self.square_loss:
          pos_loss += F.relu(torch.mean((torch.mean(feature_set, dim=0) - feature_set).pow(2).sum(-1)) - self.pos_thresh)
        else:
          pos_loss += F.relu(torch.mean(torch.sqrt((torch.mean(feature_set, dim=0) - feature_set).pow(2).sum(-1) + 1e-7)) - self.pos_thresh)
      # whether we should block the gradient at the finest position during loss calculation
      if self.block_finest_gradient:
        blocked_feature_set = feature_set[torch.bitwise_not(finest_flag_set)]
        finest_loss += F.relu(torch.sqrt((torch.mean(blocked_feature_set, dim=0) - 
            feature_set[finest_flag_set][0].detach()).pow(2).sum() + 1e-7) - self.finest_thresh)
      else:
        if self.square_loss:
          finest_loss += F.relu((torch.mean(feature_set, dim=0) - 
              feature_set[finest_flag_set][0]).pow(2).sum() - self.finest_thresh)
        else:
          finest_loss += F.relu(torch.sqrt((torch.mean(feature_set, dim=0) - 
              feature_set[finest_flag_set][0]).pow(2).sum() + 1e-7) - self.finest_thresh)
      # if self.block_finest_gradient:
      #   blocked_feature_set = feature_set[torch.bitwise_not(finest_flag_set)]
      #   finest_loss += torch.mean(F.relu(torch.sqrt((blocked_feature_set - 
      #       feature_set[finest_flag_set][0].detach()).pow(2).sum(-1) + 1e-7) - self.finest_thresh))
      # else:
      #   if self.square_loss:
      #     finest_loss += torch.mean(F.relu((feature_set - 
      #         feature_set[finest_flag_set][0]).pow(2).sum(-1) - self.finest_thresh))
      #   else:
      #     finest_loss += torch.mean(F.relu(torch.sqrt((feature_set - 
      #         feature_set[finest_flag_set][0]).pow(2).sum(-1) + 1e-7) - self.finest_thresh))
    pos_loss, finest_loss = pos_loss/len(pos_sel), finest_loss/len(pos_sel)
    pos_timer.toc()

    # negative loss
    dist_timer.tic()
    # calculate between two bunches of separately downsampled features
    sel_hn1 = np.random.choice(N_out, min(N_out, max_hn_samples), replace=False)
    sel_hn2 = np.random.choice(N_out, min(N_out, max_hn_samples), replace=False)
    subF1 = F_out[sel_hn1]
    subF2 = F_out[sel_hn2]
    D_fs = pdist(subF1, subF2, dist_type='L2')
    if self.use_hard_negative:
      D_fs_min, D_fs_ind = D_fs.min(1)
    else:
      D_fs_ind = torch.Tensor([np.random.choice(D_fs.shape[1], 1) for _ in range(D_fs.shape[0])]).long().to(D_fs.device)
      D_fs_min = D_fs[torch.arange(D_fs.shape[0]), D_fs_ind]
    D_fs_ind = D_fs_ind.cpu()
    dist_timer.toc()

    # mask the equal-indexed negative to prevent self comparison
    neg_timer.tic()
    mask_self = (sel_hn1 != sel_hn2[D_fs_ind])

    sel_hn2_closest = sel_hn2[D_fs_ind]
    hash2_timer.tic()
    pos_keys = index_hash
    neg_keys = _neg_hash(sel_hn1, sel_hn2_closest, hash_seed)
    hash2_timer.toc()

    mask = np.logical_not(np.isin(neg_keys, pos_keys, assume_unique=False))
    neg_loss = F.relu(self.neg_thresh - D_fs_min[mask & mask_self]).pow(2)
    neg_timer.toc()

    # print(f"split time: {split_timer.avg}, pos time: {pos_timer.avg}, dist time: {dist_timer.avg}, " +
    #       f"hash_timer: {hash2_timer.avg}, neg time: {neg_timer.avg}")
    return pos_loss, finest_loss, neg_loss.mean()


  def location_circle_loss(self,
                           F_out,
                           group,
                           index,
                           index_hash,
                           finest_flag,
                           max_pos_cluster=256,
                           max_hn_samples=None,
                           points=None,
                           batch_lengths=None):
    """
    Calculates the finest and circle losses of input co-location groups.
    Works only with proper data input from colocation_data_loader.
    """
    group, index = group.to(self.device), index.to(self.device)
    split_timer, pos_timer, dist_timer, neg_timer = Timer(), Timer(), Timer(), Timer()

    # split the groups, and then select some of them to calculate loss
    split_timer.tic()
    pos_loss, finest_loss = 0, 0
    N_pos_clusters = len(group)
    index_split = torch.split(index, tuple(group.tolist()))
    finest_flag_split = torch.split(finest_flag, tuple(group.tolist()))
    if N_pos_clusters > max_pos_cluster:
      pos_sel = np.sort(np.random.choice(N_pos_clusters, max_pos_cluster, replace=False))
    else:
      pos_sel = np.arange(N_pos_clusters)
    split_timer.toc()

    # create a place for storing the selected positive features and positive coordinates
    coords_sel = torch.zeros((len(pos_sel), 3)).float().to(self.device)
    feats_sel = torch.zeros((len(pos_sel), F_out.shape[1])).float().to(self.device)

    # we count the number of positive anchors for all Point Clouds in a batch in 'batch_pos_lengths'.
    # this is then used to calculate a batch mask. It masks all in-batch, cross-item negative loss calculation
    batch_size = len(batch_lengths)
    accumulated_batch_lengths = np.zeros(batch_size)
    accumulated_batch_lengths[0] = batch_lengths[0]
    for i, _ in enumerate(batch_lengths):
      if i > 0:
        accumulated_batch_lengths[i] = batch_lengths[i] + accumulated_batch_lengths[i-1]
    batch_pos_lengths = np.zeros(batch_size).astype(int)

    # positive loss and finest loss
    pos_timer.tic()
    for count, i in enumerate(pos_sel):
      index_set, finest_flag_set = index_split[i], finest_flag_split[i]
      coords_sel[count] = points[index_set[0]]
      feature_set = F_out[index_set]
      feats_sel[count] = torch.mean(feature_set, dim=0)

      # record #pos for all elements in a batch to construct batch_mask
      pivot = index_set[0]
      idx = np.sum(pivot.item() > accumulated_batch_lengths) # the bin index, i.e., the index inside a batch
      batch_pos_lengths[idx] += 1

      # pos_loss: choose between pair loss or variance loss
      if self.use_pair_group_positive_loss:
        # notice that the circle loss of a single pair simply degrades to softplus(feature_distance)
        pos_positions = np.random.choice(len(feature_set), 2, replace=False)
        if self.square_loss:
          pos_loss += F.softplus(
              (feature_set[pos_positions[0]] - feature_set[pos_positions[1]]).pow(2).sum(-1) -
              self.pos_thresh)
        else:
          pos_loss += F.softplus(
              torch.sqrt((feature_set[pos_positions[0]] -
                          feature_set[pos_positions[1]]).pow(2).sum(-1) + 1e-7) -
              self.pos_thresh)
      else:
        # pos_thresh is divided by 2, so that arbitrary two features will have at most pos_thresh distance
        if self.square_loss:
          var_dists = (torch.mean(feature_set, dim=0) - feature_set).pow(2).sum(-1)  - self.pos_thresh / 2
        else:
          var_dists = torch.sqrt(
              (torch.mean(feature_set, dim=0) - feature_set).pow(2).sum(-1) +
              1e-7) - self.pos_thresh / 2
        pos_loss += F.softplus(
            torch.logsumexp(
                self.log_scale * var_dists *
                torch.max(torch.zeros_like(var_dists), var_dists).detach(),
                dim=-1)) / self.log_scale

      # finest_loss: whether we should block the gradient at the finest position during loss calculation
      if self.block_finest_gradient:
        blocked_feature_set = feature_set[~finest_flag_set]
        if self.square_loss:
          finest_dists = (blocked_feature_set - feature_set[finest_flag_set][0].detach() ).pow(2).sum(-1)- self.finest_thresh
        else:
          finest_dists = torch.sqrt(
              (blocked_feature_set - feature_set[finest_flag_set][0].detach()
              ).pow(2).sum(-1) + 1e-7) - self.finest_thresh
      else:
        if self.square_loss:
          finest_dists = (feature_set - feature_set[finest_flag_set][0]).pow(2).sum(-1) - self.finest_thresh
        else:
          finest_dists = torch.sqrt(
              (feature_set - feature_set[finest_flag_set][0]).pow(2).sum(-1) +
              1e-7) - self.finest_thresh
      finest_loss += F.softplus(
          torch.logsumexp(
              self.log_scale * finest_dists *
              torch.max(torch.zeros_like(finest_dists), finest_dists).detach(),
              dim=-1)) / self.log_scale

    pos_loss, finest_loss = pos_loss / len(pos_sel), finest_loss / len(pos_sel)
    pos_timer.toc()

    # build batch_mask
    batch_mask = torch.zeros((len(pos_sel), len(pos_sel))).bool().to(self.device)
    start_ind = 0
    for i in range(batch_size):
      # create diagonally concatenated all-one matrix as the mask
      batch_mask[start_ind:start_ind+batch_pos_lengths[i], start_ind:start_ind+batch_pos_lengths[i]] = True
      start_ind += batch_pos_lengths[i]

    # negative loss
    dist_timer.tic()
    # get L2 coords & feature distance between groups
    coords_dist = torch.sqrt(
        square_distance(coords_sel[None, :, :], coords_sel[None, :, :]).squeeze(0))
    feats_dist = torch.sqrt(
        square_distance(feats_sel[None, :, :], feats_sel[None, :, :],
                        normalised=True)).squeeze(0)
    dist_timer.toc()

    # calculate negative circle loss
    neg_timer.tic()
    neg_mask = (coords_dist > self.safe_radius) & batch_mask # mask the close anchors and not-the-same-input anchors
    sel = (neg_mask.sum(-1) > 0).detach()  # Find anchors that have negative pairs. All anchors naturally have positive pairs.

    neg_weight = feats_dist + 1e5 * (~neg_mask).float()  # mask the non-negative (self-comparison is also removed here)
    neg_weight = (self.neg_thresh - neg_weight)  # mask the uninformative negative
    neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach()
    lse_neg_row = torch.logsumexp(
        self.log_scale * (self.neg_thresh - feats_dist) * neg_weight, dim=-1)
    loss_row = F.softplus(lse_neg_row) / self.log_scale

    neg_loss = loss_row[sel].mean()
    neg_timer.toc()

    # print(f"split time: {split_timer.avg}, pos time: {pos_timer.avg}, dist time: {dist_timer.avg}, " +
    #       f"neg time: {neg_timer.avg}")
    return pos_loss, finest_loss, neg_loss

  def _get_dist_err_plot(self,
                         F_out,
                         group,
                         index,
                         central_distance,
                         group_batch_lengths,
                         finest_flag,
                         max_pos_cluster=512,
                         mean=False,
                         return_pair_dist=False):
    '''
    Calculates for each colocation group its distance-feature_err relation.
    '''
    N_pos_clusters = len(group)
    index_split = torch.split(index, tuple(group.tolist()))
    finest_flag_split = torch.split(finest_flag, tuple(group.tolist()))
    central_distance_split = torch.split(central_distance, tuple(group.tolist()))

    if N_pos_clusters > max_pos_cluster:
      pos_sel = np.random.choice(N_pos_clusters, max_pos_cluster, replace=False)
      pos_sel.sort()
    else:
      pos_sel = np.arange(N_pos_clusters)

    distance_errs = []
    feature_errs = []
    pair_dists = []
    for i in pos_sel:
      index_set, finest_flag_set, central_distance_set = index_split[i], finest_flag_split[i], central_distance_split[i]
      feature_set = F_out[index_set]
      finest_distance = central_distance_set[finest_flag_set][0]

      if not mean:
        distance_errs += (central_distance_set-finest_distance).tolist()
        feature_errs += torch.norm(feature_set - feature_set[finest_flag_set][0], dim=1).tolist()
      else:
        sorted_dist, indices = torch.sort(central_distance_set)
        distance_errs += sorted_dist
        features_sorted_set = feature_set[indices]
        feature_succesive_mean = torch.Tensor([np.mean(features_sorted_set[i:].detach().cpu().numpy(), axis=0) for i in range(len(indices))]).to(feature_set.device)
        feature_errs += torch.norm(feature_succesive_mean - feature_set[finest_flag_set][0], dim=1).tolist()

      if return_pair_dist:
        pos_positions = np.random.choice(len(feature_set), 2, replace=False)
        pair_dists.append(central_distance_set[pos_positions])

    group_accumu_lengths = np.cumsum(group_batch_lengths.tolist())
    group_inbatch_indexes = [np.where(idx <= group_accumu_lengths)[0][0] for idx in pos_sel]
    return distance_errs, feature_errs, pair_dists, group_inbatch_indexes


  def location_contrastive_loss(self,
                                F_out,
                                group,
                                index,
                                index_hash,
                                finest_flag,
                                max_pos_cluster=256,
                                max_hn_samples=None,
                                points=None,
                                batch_lengths=None):
    """
    Calculates the finest, positive, and negative losses of input co-location groups.
    Works only with proper data input from colocation_data_loader.
    """
    group, index = group.to(self.device), index.to(self.device)
    N_out = len(F_out)
    hash_seed = N_out
    split_timer, pos_timer, dist_timer, hash2_timer, neg_timer = Timer(), Timer(), Timer(), Timer(), Timer()

    # positive loss and finest loss
    split_timer.tic()
    pos_loss = 0
    N_pos_clusters = len(group)
    index_split = torch.split(index, tuple(group.tolist()))
    if N_pos_clusters > max_pos_cluster:
      pos_sel = np.random.choice(N_pos_clusters, max_pos_cluster, replace=False)
    else:
      pos_sel = np.arange(N_pos_clusters)
    split_timer.toc()

    pos_timer.tic()
    for i in pos_sel:
      index_set = index_split[i]
      feature_set = F_out[index_set]
      if self.use_pair_group_positive_loss:
        pos_positions = np.random.choice(len(feature_set), 2, replace=False)
        pos_loss += F.relu(torch.sqrt((feature_set[pos_positions[0]] - feature_set[pos_positions[1]]).pow(2).sum(-1) + 1e-7) - self.pos_thresh)
      else:
        pos_loss += F.relu(torch.mean(torch.sqrt((torch.mean(feature_set, dim=0) - feature_set).pow(2).sum(-1) + 1e-7)) - self.pos_thresh)

    pos_loss = pos_loss/len(pos_sel)
    pos_timer.toc()

    # negative loss
    dist_timer.tic()
    # calculate between two bunches of separately downsampled features
    sel_hn1 = np.random.choice(N_out, min(N_out, max_hn_samples), replace=False)
    sel_hn2 = np.random.choice(N_out, min(N_out, max_hn_samples), replace=False)
    subF1 = F_out[sel_hn1]
    subF2 = F_out[sel_hn2]
    D_fs = pdist(subF1, subF2, dist_type='L2')
    if self.use_hard_negative:
      D_fs_min, D_fs_ind = D_fs.min(1)
    else:
      D_fs_ind = torch.Tensor([np.random.choice(D_fs.shape[1], 1) for _ in range(D_fs.shape[0])]).long().to(D_fs.device)
      D_fs_min = D_fs[torch.arange(D_fs.shape[0]), D_fs_ind]
    D_fs_ind = D_fs_ind.cpu()
    dist_timer.toc()

    # mask the equal-indexed negative to prevent self comparison
    neg_timer.tic()
    mask_self = (sel_hn1 != sel_hn2[D_fs_ind])

    sel_hn2_closest = sel_hn2[D_fs_ind]
    hash2_timer.tic()
    pos_keys = index_hash
    neg_keys = _neg_hash(sel_hn1, sel_hn2_closest, hash_seed)
    hash2_timer.toc()

    mask = np.logical_not(np.isin(neg_keys, pos_keys, assume_unique=False))
    neg_loss = F.relu(self.neg_thresh - D_fs_min[mask & mask_self]).pow(2)
    neg_timer.toc()

    # print(f"split time: {split_timer.avg}, pos time: {pos_timer.avg}, dist time: {dist_timer.avg}, " +
    #       f"hash_timer: {hash2_timer.avg}, neg time: {neg_timer.avg}")
    return pos_loss, torch.Tensor([0])[0], neg_loss.mean()

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    all_dist_err = []
    all_feat_err = []
    all_pair_dist= []
    all_ingroup_indexes = []
    use_mean = False
    return_pair_dist = True

    iter_num = len(data_loader) // iter_size
    if self.calc_distance_err:
      iter_num=20
    for curr_iter in range(iter_num):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_finest_loss, batch_neg_loss, batch_loss = 0, 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        sinput = ME.SparseTensor(
            input_dict['sinput_F'].to(self.device),
            coordinates=input_dict['sinput_C'].to(self.device))
        F_out = self.model(sinput).F

        group, index, finest_flag = input_dict['group'], input_dict['index'], input_dict['finest_flag']

        pos_loss, finest_loss, neg_loss = self.location_loss(
              F_out,
              group,
              index,
              input_dict['index_hash'],
              finest_flag,
              max_pos_cluster=self.config.num_pos_per_batch * self.config.batch_size,
              max_hn_samples=self.config.num_hn_samples_per_batch * self.config.batch_size,
              points=input_dict['sinput_C'][:, 1:],
              batch_lengths=input_dict['batch_lengths'])

        if self.calc_distance_err:
          distance_errs, feature_errs, pair_dists, group_inbatch_indexes = self._get_dist_err_plot(F_out,
              group,
              index,
              input_dict['central_distance'],
              input_dict['group_lengths'],
              finest_flag,
              mean=use_mean,
              return_pair_dist=return_pair_dist)
          all_dist_err += distance_errs
          all_feat_err += feature_errs
          all_pair_dist.append(pair_dists)
          all_ingroup_indexes.append(group_inbatch_indexes)

        pos_loss /= iter_size
        finest_loss /= iter_size
        neg_loss /= iter_size
        loss = self.pos_weight * pos_loss + self.finest_weight * finest_loss + self.neg_weight * neg_loss
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_finest_loss += finest_loss.item()
        batch_neg_loss += neg_loss.item()

      if not self.calc_distance_err:
        self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f} Finest: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss, batch_finest_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

    if self.calc_distance_err:
      tail = "normal" if (not use_mean) else "mean"
      np.savez(self.checkpoint_dir+f"/dist_err_{tail}", distance=all_dist_err, err=all_feat_err, 
          pair_dist=all_pair_dist, inbatch_index=all_ingroup_indexes)
      print("Saved distance-err points!", flush=True)
      raise ValueError


class TripletLossTrainer(ContrastiveLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=None,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()

    return loss, pos_dist.mean(), rand_neg_dist.mean()

  def _train_epoch(self, epoch):
    config = self.config

    gc.collect()
    self.model.train()

    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    pos_dist_meter, neg_dist_meter = AverageMeter(), AverageMeter()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_loss = 0
      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'].to(self.device),
            coordinates=input_dict['sinput0_C'].to(self.device))
        F0 = self.model(sinput0).F

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'].to(self.device),
            coordinates=input_dict['sinput1_C'].to(self.device))
        F1 = self.model(sinput1).F

        pos_pairs = input_dict['correspondences']
        loss, pos_dist, neg_dist = self.triplet_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=config.triplet_num_pos * config.batch_size,
            num_hn_samples=config.triplet_num_hn * config.batch_size,
            num_rand_triplet=config.triplet_num_rand * config.batch_size)
        loss /= iter_size
        loss.backward()
        batch_loss += loss.item()
        pos_dist_meter.update(pos_dist)
        neg_dist_meter.update(neg_dist)

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e}, Pos dist: {:.3e}, Neg dist: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, pos_dist_meter.avg, neg_dist_meter.avg) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        pos_dist_meter.reset()
        neg_dist_meter.reset()
        data_meter.reset()
        total_timer.reset()


class HardestTripletLossTrainer(TripletLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=512,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(
        torch.cat([
            rand_pos_dist + self.neg_thresh - rand_neg_dist,
            pos_dist[mask0] + self.neg_thresh - D01min[mask0],
            pos_dist[mask1] + self.neg_thresh - D10min[mask1]
        ])).mean()

    return loss, pos_dist.mean(), (D01min.mean() + D10min.mean()).item() / 2
