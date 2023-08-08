import open3d as o3d  # prevent loading error

import sys
import logging
import json
import argparse
import numpy as np
from easydict import EasyDict as edict

import torch
from model import load_model
from scripts.SC2_PCR.SC2_PCR import Matcher

from lib.complement_data_loader import make_data_loader
from lib.eval import find_nn_gpu

from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
from lib.timer import AverageMeter, Timer

import MinkowskiEngine as ME
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

def find_corr(xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=500)
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

def evaluate_nn_dist(xyz0, xyz1, T_gth):
    xyz0 = apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return dist.tolist()

def random_sample(pcd, feats, N):
    """
    Do random sampling to get exact N points and associated features
    pcd:    [N,3]
    feats:  [N,C]
    """
    if(isinstance(pcd,torch.Tensor)):
        n1 = pcd.size(0)
    elif(isinstance(pcd, np.ndarray)):
        n1 = pcd.shape[0]

    if n1 == N:
        return pcd, feats

    if n1 > N:
        choice = np.random.permutation(n1)[:N]
    else:
        choice = np.random.choice(n1, N)

    return pcd[choice], feats[choice]

def main(config):
  test_loader = make_data_loader(
      config, config.test_phase, 1, num_threads=config.test_num_thread, shuffle=False)

  num_feats = 1

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  Model = load_model(config.model)
  model = Model(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      conv1_kernel_size=config.conv1_kernel_size,
      normalize_feature=config.normalize_feature)
  checkpoint = torch.load(config.save_dir + '/best_val_checkpoint.pth')
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device)
  model.eval()

  use_sc2pcr = not config.use_RANSAC
  if use_sc2pcr:
    matcher = Matcher(inlier_threshold=config.inlier_threshold,
                  num_node=config.num_node,
                  use_mutual=config.use_mutual,
                  d_thre=config.d_thre,
                  num_iterations=config.num_iterations,
                  ratio=config.ratio,
                  nms_radius=config.nms_radius,
                  max_points=config.max_points,
                  k1=config.k1,
                  k2=config.k2, )

  success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
  data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()

  # test_loader.dataset.__getitem__(610)

  test_iter = test_loader.__iter__()
  N = len(test_iter)
  n_gpu_failures = 0

  # downsample_voxel_size = 2 * config.voxel_size
  dists_success = []
  dists_fail = []
  dists_nn = []
  list_rte = []
  list_rre = []
  trans_gt = []
  T_gt = []
  T_est = []

  # rte_thresh = 0.6
  # rre_thresh = 1.5

  rte_thresh = config.rte_thresh
  rre_thresh = config.rre_thresh
  print(f"rre thresh: {rre_thresh}; rte_thresh: {rte_thresh}")

  # for i in range(100):
  for i in range(len(test_iter)):
    data_timer.tic()
    # for _ in range(207):
    #   data_dict = test_iter.next() 
    #   print(f"fast loaded {i}'th data", flush=True)
    #   i = i+1
    data_dict = test_iter.next()
    # try:
    #   data_dict = test_iter.next()
    # except ValueError:
    #   n_gpu_failures += 1
    #   logging.info(f"# Erroneous GPU Pair {n_gpu_failures}")
    #   continue
    data_timer.toc()
    xyz0, xyz1 = data_dict['pcd0'][0], data_dict['pcd1'][0]
    T_gth = data_dict['T_gt']
    T_gt.append(T_gth)
    dist_gth = np.sqrt(np.sum((T_gth[:3, 3].cpu().numpy())**2))
    trans_gt.append(dist_gth)
    xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

    # pcd0 = make_open3d_point_cloud(xyz0np)
    # pcd1 = make_open3d_point_cloud(xyz1np)

    # torch.cuda.synchronize()
    with torch.no_grad():
      feat_timer.tic()
      sinput0 = ME.SparseTensor(
          data_dict['sinput0_F'].to(device), coordinates=data_dict['sinput0_C'].to(device))
      enc0 = model(sinput0)
      F0 = enc0.F.detach()
      sinput1 = ME.SparseTensor(
          data_dict['sinput1_F'].to(device), coordinates=data_dict['sinput1_C'].to(device))
      enc1 = model(sinput1)
      F1 = enc1.F.detach()
      # torch.cuda.synchronize()
      feat_timer.toc()

    xyz0_corr, xyz1_corr = find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
    # if i==207:
    #   np.savez(f'corr_fcgf', src=xyz0.cpu().detach().numpy(), tgt=xyz1.cpu().detach().numpy(),
    #           corr_src=xyz0_corr, corr_tgt=xyz1_corr, 
    #           T_gt=T_gth.cpu().detach().numpy())
    # raise ValueError
    dists_nn.append(evaluate_nn_dist(xyz0_corr, xyz1_corr, T_gth))
    
    # i=0
    # for enc, pcd in zip([enc0, enc1], [pcd0, pcd1]):
    #   _, batch_enc_feats = enc.detach().decomposed_coordinates_and_features
    #   # print(np.size(batch_enc_feats))
    #   from sklearn.manifold import TSNE
    #   tsne = TSNE(n_components=3)
    #   print('doing tsne')
    #   X_embedded = tsne.fit_transform(batch_enc_feats[0].cpu().numpy())
    #   print('finished')
    #   xyz = np.asarray(pcd.points)
    #   inds = np.where(xyz[:, 2] < 0)
    #   bag = np.hstack((xyz, X_embedded))
    #   bag = np.delete(bag, inds, 0)
    #   xyz = bag[:, :3]
    #   X_embedded = bag[:, 3:]
    #   for dim in range(3):
    #     max_value = np.max(X_embedded[:, dim])
    #     min_value = np.min(X_embedded[:, dim])
    #     X_embedded[:, dim] = (X_embedded[:, dim] - min_value) / (max_value - min_value)
    #   np.save(f'points{i}.npy', np.asarray(pcd.points))
    #   np.save(f'colors{i}.npy', np.asarray(pcd.colors))
    #   i+=1
    # raise ValueError

    n_points = 5000
    ########################################
    # run random sampling or probabilistic sampling
    xyz0np, F0 = random_sample(xyz0np, F0, n_points)
    xyz1np, F1 = random_sample(xyz1np, F1, n_points)

    pcd0 = make_open3d_point_cloud(xyz0np)
    pcd1 = make_open3d_point_cloud(xyz1np)

    feat0 = make_open3d_feature(F0, config.model_n_out, F0.shape[0])
    feat1 = make_open3d_feature(F1, config.model_n_out, F1.shape[0])

    reg_timer.tic()
    distance_threshold = config.voxel_size * 1.0
    if not use_sc2pcr:
      ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
          pcd0, pcd1, feat0, feat1, False, distance_threshold,
          o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
          ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 10000))
      T_ransac = torch.from_numpy(ransac_result.transformation.astype(np.float32))
    else:
      # result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
      #     pcd0, pcd1, feat0, feat1, False, distance_threshold,
      #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
      #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
      #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
      #     ],o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
      # T_ransac = torch.from_numpy(result_ransac.transformation.astype(np.float32))

      xyz0, xyz1 = torch.from_numpy(xyz0np).to(F0.device), torch.from_numpy(xyz1np).to(F0.device)
      T_ransac, _, _, _ = matcher.estimator(xyz0[None,:], xyz1[None,:], F0[None,:], F1[None,:])
      T_ransac = T_ransac[0].to("cpu")

    reg_timer.toc()

    T_est.append(T_ransac)

    # Translation error
    rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
    trace_matrix = T_ransac[:3, :3].t() @ T_gth[:3, :3]    # patch for numerical stability. When Trace(trace_matrix) > 3, this fix will eliminate nan when calculating arccos
    trace_matrix[[0,1,2], [0,1,2]] = torch.min(torch.ones(3), trace_matrix[[0,1,2], [0,1,2]])
    rre = np.arccos((np.trace(trace_matrix) - 1) / 2)

    # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
    # http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf
    # rte_thresh = 0.6
    # rre_thresh = 1.5
    
    if rte < rte_thresh:
      rte_meter.update(rte)

    if not np.isnan(rre) and rre < np.pi / 180 * rre_thresh:
      rre_meter.update(rre * 180 / np.pi)

    if rte < rte_thresh and not np.isnan(rre) and rre < np.pi / 180 * rre_thresh:
      success_meter.update(1)
      dists_success.append(dist_gth)
    else:
      success_meter.update(0)
      dists_fail.append(dist_gth)
      logging.info(f"Failed with RTE: {rte}, RRE: {rre * 180 / np.pi}")

    list_rte.append(rte)
    list_rre.append(rre)

    if i % 10 == 0:
      logging.info(
          f"{i} / {N}: Data time: {data_timer.avg}, Feat time: {feat_timer.avg}," +
          f" Reg time: {reg_timer.avg}, RTE: {rte_meter.avg}," +
          f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}"
          + f" ({success_meter.avg * 100} %)")
      # data_timer.reset()
      # feat_timer.reset()
      # reg_timer.reset()

  print(f"rre thresh: {rre_thresh}; rte_thresh: {rte_thresh}")

  # np.savez(f"results_fcgf_LoNuScenes.npz", rres=list_rre, rtes=list_rte, dists=trans_gt, T_gt=T_gt, T_est=T_est)

  # dists_nn = np.array(dists_nn, dtype=object) # dists_nn: N_pairs * N_num_point_one_frame
  # hit_ratios = []
  # for seed_tao_1 in range(1, 11):
  #   tao_1 = seed_tao_1 * 0.1
  #   hit_ratios.append(np.mean((dists_nn < tao_1).astype(float), axis=1).tolist())

  # fmrs = []
  # # hit_ratios: 10 * N_pairs
  # for seed_tao_2 in range(1, 11):
  #   tao_2 = seed_tao_2 * 0.01
  #   fmrs.append(np.mean((np.array(hit_ratios) > tao_2).astype(float), axis=1).tolist())
  
  # fmrs: 10*10, dim0: seed_tao_2; dim2: seed_tao_1
  # np.save('FMRs_fcgf.npy', fmrs)

  # if config.use_old_pose:
  #   np.savez(f"results_fcgf_{config.pair_max_dist}.npz", rres=list_rre, rtes=list_rte, dists=trans_gt, T_gt=T_gt, T_est=T_est)
  # else:
  # np.savez(f"results_fcgf_nuScenes_{config.pair_max_dist}.npz", rres=list_rre, rtes=list_rte, dists=trans_gt, T_gt=T_gt, T_est=T_est)

  # np.save("success_dists.npy", dists_success)
  # np.save("fail_dists.npy", dists_fail)

  # np.save("list_rte_fcgf.npy", list_rte)
  # np.save("list_rre_fcgf.npy", list_rre)

  logging.info(
      f"RTE: {rte_meter.avg}, var: {rte_meter.var}," +
      f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
      f"/ {success_meter.count} ({success_meter.avg * 100} %)")


def str2bool(v):
  return v.lower() in ('true', '1')
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', default=None, type=str)
  parser.add_argument('--test_phase', default='test', type=str)
  parser.add_argument('--LoKITTI', default=False, type=str2bool)
  parser.add_argument('--LoNUSCENES', default=False, type=str2bool)
  parser.add_argument('--DAIR_V2X', default=False, type=str2bool)
  parser.add_argument('--test_num_thread', default=5, type=int)
  parser.add_argument('--pair_min_dist', default=None, type=int)
  parser.add_argument('--pair_max_dist', default=None, type=int)
  parser.add_argument('--downsample_single', default=1.0, type=float)
  parser.add_argument('--kitti_root', type=str, default="/data/kitti/")
  parser.add_argument('--use_RANSAC', type=str2bool, default=True)
  parser.add_argument('--rre_thresh', default=5.0, type=float)
  parser.add_argument('--rte_thresh', default=2.0, type=float)
  args = parser.parse_args()

  config = json.load(open(args.save_dir + '/config.json', 'r'))
  config = edict(config)
  config.save_dir = args.save_dir
  config.test_phase = args.test_phase
  config.kitti_root = args.kitti_root
  config.kitti_odometry_root = args.kitti_root + '/dataset'
  config.test_num_thread = args.test_num_thread
  config.LoKITTI = args.LoKITTI
  config.LoNUSCENES = args.LoNUSCENES
  config.debug_use_old_complement = True
  config.debug_need_complement = False
  config.debug_manual_seed = True
  config.phase = 'test'
  config.use_RANSAC = args.use_RANSAC
  # config.debug_force_icp_recalculation = True

  if args.DAIR_V2X:
    config.dataset = "VICRegistrationDataset"
    config.test_phase = "val"

  if args.LoNUSCENES:
    config.dataset = "PairComplementNuscenesDataset"
    config.use_old_pose = True
  
  if args.LoKITTI:
    config.dataset = "PairComplementKittiDataset"

  if not config.use_RANSAC:
    # merge SC2-PCR configs
    config_sc2pcr = json.load(open('scripts/SC2_PCR/config_json/config_KITTI.json', 'r'))
    config_sc2pcr = edict(config_sc2pcr)
    for key, item in config_sc2pcr.items():
      config[key] = item

  if args.pair_min_dist is not None and args.pair_max_dist is not None:
    config.pair_min_dist = args.pair_min_dist
    config.pair_max_dist = args.pair_max_dist
  config.downsample_single = args.downsample_single

  config.rte_thresh = args.rte_thresh
  config.rre_thresh = args.rre_thresh

  main(config)
