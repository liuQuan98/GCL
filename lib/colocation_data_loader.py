# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm
import pathlib
from functools import partial

from util.pointcloud import get_matching_indices_colocation_simple, get_matching_indices_colocation, make_open3d_point_cloud
import lib.transforms as t
from lib.timer import Timer, AverageMeter
from scipy.spatial.transform import Rotation

import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from util.misc import _exhaustive_hash

import MinkowskiEngine as ME

import open3d as o3d

kitti_cache = {}
kitti_icp_cache = {}


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

def follow_presampled_trans(pcd, trans):
    T = np.eye(4)
    R = trans[:3, :3]
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

class PointDataset(torch.utils.data.Dataset):
    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        self.phase = phase
        self.files = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.random_dist = True
        if 'random_dist' in [k for (k, v) in config.items()]:
            self.random_dist = config.random_dist
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        trans = trans.astype(np.float32)
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.files)

class KittiDataset(PointDataset):
    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                        '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    def get_video_odometry(self, dirname, indices=None, ext='.txt', return_all=False):
        if type(dirname) == int or type(dirname) == np.int64: # kitti
            data_path = self.root + '/poses/%02d.txt' % dirname
            if data_path not in kitti_cache:
                kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return kitti_cache[data_path]
            else:
                return kitti_cache[data_path][indices]
        else:   # nuscenes
            data_path = os.path.join(self.root, 'sequences', dirname, 'poses.npy')
            if data_path not in self.nuscenes_cache:
                self.nuscenes_cache[data_path] = np.load(data_path)
            if return_all:
                return self.nuscenes_cache[data_path]
            else:
                return self.nuscenes_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0


class ColocationKittiDataset(KittiDataset):
    '''
    Training phase dataloader that loads a point cloud and a random neighborhood.
    Only compatible during training phase, and should be used with Colocation-Contrastive Loss.
    '''
    
    DATA_FILES = {
        'train': './config/train_kitti.txt',
        'val': './config/val_kitti.txt',
        'test': './config/test_kitti.txt'
    }
    
    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PointDataset.__init__(self, phase, transform, random_rotation, 
                              random_scale, manual_seed, config)

        self.root = root = config.kitti_root + '/dataset'
        self.matching_search_voxel_size = \
            config.voxel_size * config.positive_pair_search_voxel_size_multiplier

        self.calc_distance_err = config.calc_distance_err

        self.MIN_DIST = config.min_dist
        self.MAX_DIST = config.max_dist
        self.num_neighborhood = config.num_neighborhood
        assert self.num_neighborhood % 2 == 0, "Parameter 'num_neighborhood' must be even!"

        if config.voxel_size < 0.2:
            self.max_in_p = config.max_in_p
        else:
            self.max_in_p = 1e7  # no random discarding is performed when voxel size is large

        self.icp_path = os.path.join(config.kitti_root,'icp_slam')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        logging.info(f"Loading the subset {phase} from {root}")
        self.phase=phase
        assert self.phase == 'train', "Colocation Data Loader loads a point cloud and its neighbourhood, which is only meaningful during training time!"

        self.area_length_per_neighbor = 2*self.MAX_DIST / self.num_neighborhood

        # this assertion ensures the inner area can spawn a neighborhood point cloud
        assert self.MIN_DIST < self.area_length_per_neighbor, "MIN_DIST is too high compared to area_length_per_neighbor! Lower MIN_DIST or lower num_neighborhood instead."
        self.config = config

        self.prepare_kitti_ply_colocation(phase)
        print(f"Data size for phase {phase}: {len(self.files)}")

    def prepare_kitti_ply_colocation(self, phase):
        # load all frames that have a full spatial neighbourhood
        subset_names = open(self.DATA_FILES[phase]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            print(f"Processing drive {drive_id}")
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_pos = self.get_slam_odometry(drive_id, return_all=True)
            self.Ts = all_pos[:, :3, 3]

            # skip first several frames, since they do not have a full neighborhood
            curr_time = inames[min(int(self.MAX_DIST * 5), int(len(inames)/2))]

            np.random.seed(0)
            while curr_time in inames:
                # find the current neighborhood
                skip, nghb = self._get_neighborhood_frames(curr_time)

                if skip:
                    curr_time += 1
                else:
                    self.files.append((drive_id, curr_time, nghb))
                    curr_time += 11 # empirical distance parameter between centers


    def parse_calibration(self, filename):
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()
        return calib
    
    def get_slam_odometry(self, drive, indices=None, return_all=False):
        data_path = self.root + '/sequences/%02d' % drive
        calib_filename = data_path + '/calib.txt'
        pose_filename = data_path + '/poses.txt'
        calibration = self.parse_calibration(calib_filename)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []
        pose_file = open(pose_filename)
        for line in pose_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        
        if pose_filename not in kitti_icp_cache:
            kitti_icp_cache[pose_filename] = np.array(poses)
        if return_all:
            return kitti_icp_cache[pose_filename]
        else:
            return kitti_icp_cache[pose_filename][indices]

    def _get_neighborhood_frames(self, frame):
        # list of frame ids belonging to the neighbourhood of the current frame
        list_complement = []
        # indicates that there aren't enough complement frames around this frame
        # so that we should skip this frame
        skip_flag = False
        # Find the frames behind me
        left_frame_bound = max(0, frame-int(10*self.MAX_DIST))
        left_dist = (self.Ts[left_frame_bound:frame] - self.Ts[frame].reshape(1, 3))**2
        left_dist = np.sqrt(left_dist.sum(-1))
        for i in range(int(self.num_neighborhood / 2)):
            area_range_min = max(self.MIN_DIST, self.area_length_per_neighbor*i)
            area_range_max = max(self.MIN_DIST, self.area_length_per_neighbor*(i+1))
            dist_tmp = area_range_min + np.random.rand() * (area_range_max - area_range_min)
            candidates = np.where(left_dist > dist_tmp)[0]
            # print(candidates)
            if len(candidates) == 0:
                # No left-side complement detected
                skip_flag = True
                break
            else:
                list_complement.append(left_frame_bound + candidates[-1])
        
        if skip_flag:
            return (True, [])

        # Find the frames in front of me   
        right_dist = (self.Ts[frame: frame+int(10*self.MAX_DIST)] - self.Ts[frame].reshape(1, 3))**2
        right_dist = np.sqrt(right_dist.sum(-1))
        for i in range(int(self.num_neighborhood / 2)):
            area_range_min = max(self.MIN_DIST, self.area_length_per_neighbor*i)
            area_range_max = max(self.MIN_DIST, self.area_length_per_neighbor*(i+1))
            dist_tmp = area_range_min + np.random.rand() * (area_range_max - area_range_min)
            candidates = np.where(right_dist > dist_tmp)[0]
            if len(candidates) == 0:
                # No right-side complement detected
                skip_flag = True
                list_complement = []
                break
            else:
                list_complement.append(frame + candidates[0])
        return (skip_flag, list_complement)

    def _get_velodyne_fn(self, drive, t):
        return self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)

    def _get_labels_fn(self, drive, t):
        return self.root + '/sequences/%02d/labels/%06d.label' % (drive, t)

    # simple function for getting the xyz point-cloud w.r.t drive and time
    def _get_xyz(self, drive, time):
        fname = self._get_velodyne_fn(drive, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]

    # backup function used to load semantic labels for future visualization
    def _get_semantic_label(self, drive, time):
        fname = self._get_labels_fn(drive, time)
        labels = np.fromfile(fname, dtype=np.int16).reshape(-1, 2)
        return labels[:,0]

    def __getitem__(self, idx):
        prepare_timer, icp_timer, rot_timer, nn_timer = Timer(), Timer(), Timer(), Timer()
        prepare_timer.tic()
        drive, t, t_cmpl = self.files[idx]
        
        positions = self.get_slam_odometry(drive, [t] + t_cmpl)
        pos = positions[0]
        pos_cmpl = positions[1:]

        # load center point cloud
        xyz = self._get_xyz(drive, t)
        # Randomly drop some points (augmentation process and safety for GPU memory consumption)
        if xyz.shape[0] > self.max_in_p:
            xyz_inds = np.random.choice(xyz.shape[0], size=self.max_in_p, replace=False)
            xyz = xyz[xyz_inds]

        # load neighbourhood point clouds
        xyz_cmpl = []
        for t_tmp in t_cmpl:
            xyz_t = self._get_xyz(drive, t_tmp)
            if xyz_t.shape[0] > self.max_in_p:
                xyz_t_inds = np.random.choice(xyz_t.shape[0], size=self.max_in_p, replace=False)
                xyz_t = xyz_t[xyz_t_inds]
            xyz_cmpl.append(xyz_t)
        prepare_timer.toc()

        icp_timer.tic()
        # use semantic kitti label as GT transformation between center frame and its neighbors (acquired using slam)
        def GetListM(pos_core, pos_cmpls):
            return [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(0, int(self.num_neighborhood/2))] + \
                    [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(int(self.num_neighborhood/2), len(pos_cmpls))]
        list_M = GetListM(pos, pos_cmpl)
        icp_timer.toc()
        
        # add random rotation if needed, note that the same rotation is applied to both curr and nghb
        rot_timer.tic()
        if self.random_rotation:
            T0 = sample_random_trans(xyz, self.randg, np.pi / 4)

            xyz = self.apply_transform(xyz, T0)
            for i, xyz_tmp in enumerate(xyz_cmpl):
                Tc = follow_presampled_trans(xyz_tmp, T0)
                xyz_cmpl[i] = self.apply_transform(xyz_tmp, Tc)
                list_M[i] = T0 @ list_M[i] @ np.linalg.inv(Tc)

        # random scaling
        matching_search_voxel_size = self.matching_search_voxel_size
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz = scale * xyz
            for i, xyz_tmp in enumerate(xyz_cmpl):
                xyz_cmpl[i] = scale * xyz_tmp
                list_M[i][:3, 3] = scale * list_M[i][:3, 3]
        rot_timer.toc()

        # voxelization
        nn_timer.tic()
        xyz = torch.from_numpy(xyz)
        for i in range(len(xyz_cmpl)):
            xyz_cmpl[i] = torch.from_numpy(xyz_cmpl[i])

        # Make voxelized center points and voxelized center PC
        _, sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
        xyz_th = xyz[sel]
        pcd = make_open3d_point_cloud(xyz[sel])

        # Make both voxelized-unaligned nghb, and PCs
        # They are voxelized with the same index set (sel_nghb) so that both are synonimous w.r.t. point index
        pcd_cmpl = []
        xyz_cmpl_th = []
        for i in range(len(xyz_cmpl)):
            _, sel_nghb = ME.utils.sparse_quantize(xyz_cmpl[i] / self.voxel_size, return_index=True)
            xyz_cmpl_th.append(xyz_cmpl[i][sel_nghb].float())
            pcd_cmpl.append(make_open3d_point_cloud(xyz_cmpl[i][sel_nghb]))
        del sel, sel_nghb

        # Get matches
        group, index, finest_flag, central_distance = get_matching_indices_colocation(pcd, pcd_cmpl, xyz_cmpl_th, list_M, matching_search_voxel_size, self.calc_distance_err, K=5)
        nn_timer.toc()

        group = torch.Tensor(group)
        index = torch.Tensor(index)
        finest_flag = torch.Tensor(finest_flag)
        coords = torch.floor(xyz_th / self.voxel_size).int()
        feats = torch.ones((len(coords), 1)).float()

        if self.calc_distance_err:
            central_distance = torch.Tensor(central_distance)
        else:
            central_distance = torch.Tensor([0])

        coords_cmpl = []
        feats_cmpl = []
        for xyz_tmp_th in xyz_cmpl_th:
            coords_cmpl.append(torch.floor(xyz_tmp_th / self.voxel_size).int())
            feats_cmpl.append(torch.ones((len(coords_cmpl[-1]), 1)).float())

        if self.transform:
            coords, feats = self.transform(coords, feats)
        # print(f"Data loading time: prepare: {prepare_timer.avg}, icp: {icp_timer.avg}, " +
        #       f"rotate & scale: {rot_timer.avg}, nn-search: {nn_timer.avg}")

        return (xyz_th.float(), xyz_cmpl_th, 
                [coords] +  coords_cmpl, [feats] + feats_cmpl, 
                group, index, finest_flag, list_M, central_distance)


def collate_colocation_fn(list_data, config):
    xyz, xyz_nghb, coords_center_nghb, feats_center_nghb, group, index, finest_flag, list_M, central_distance = list(
        zip(*list_data))

    index_batch = []
    curr_start_ind = 0
    batch_lengths = []

    for batch_id in range(len(coords_center_nghb)):
        if len(group[batch_id]) != 0:
            index_batch.append(
                torch.from_numpy(np.array(index[batch_id]) + curr_start_ind))
        # Move the head
        curr_start_ind += np.sum([len(coords) for coords in coords_center_nghb[batch_id]])
        batch_lengths.append(np.sum([len(coords_b_tmp) for coords_b_tmp in coords_center_nghb[batch_id]]))

    coords_center_nghb_batch = []
    for coords in coords_center_nghb:
        coords_center_nghb_batch += coords
    feats_center_nghb_batch = []
    for feats in feats_center_nghb:
        feats_center_nghb_batch += feats
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_center_nghb_batch, feats_center_nghb_batch)

    # Concatenate all lists
    group_batch = torch.cat(group, 0).int()
    group_lengths = torch.Tensor([g.shape[0] for g in group])
    index_batch = torch.cat(index_batch, 0).long()
    finest_flag_batch = torch.cat(finest_flag, 0).bool()
    list_M_batch = torch.Tensor(np.concatenate(list_M, axis=0)).float()

    central_distance_batch = torch.cat(central_distance, 0).float()

    if not config.use_group_circle_loss:
        index_hash = _exhaustive_hash(torch.split(index_batch, tuple(group_batch.tolist())), len(coords_batch))
    else:
        index_hash = np.array([0])  # dummy value

    return {
        'pcd_center': xyz,
        'pcd_nghb': xyz_nghb,
        'sinput_C': coords_batch,
        'sinput_F': feats_batch.float(),
        'group': group_batch,
        'index': index_batch,
        'finest_flag': finest_flag_batch,
        'list_M': list_M_batch,
        'index_hash': index_hash,
        'central_distance': central_distance_batch,
        'batch_lengths': batch_lengths,
        'group_lengths': group_lengths
    }


class ColocationNuscenesDataset(KittiDataset):
    '''
    Training phase dataloader that loads a point cloud and a random neighborhood.
    Only compatible during training phase, and should be used with Finest-Contrastive Loss.
    '''
    
    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PointDataset.__init__(self, phase, transform, random_rotation, 
                              random_scale, manual_seed, config)
        
        self.root = root = os.path.join(config.kitti_root, phase)
        self.matching_search_voxel_size = \
            config.voxel_size * config.positive_pair_search_voxel_size_multiplier

        self.calc_distance_err = config.calc_distance_err

        self.MIN_DIST = config.min_dist
        self.MAX_DIST = config.max_dist
        self.num_neighborhood = config.num_neighborhood
        assert self.num_neighborhood % 2 == 0, "Parameter 'num_neighborhood' must be even!"

        self.icp_path = os.path.join(config.kitti_root,'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        logging.info(f"Loading the subset {phase} from {root}")
        self.phase=phase
        assert self.phase == 'train', "Colocation Data Loader loads a point cloud and its neighbourhood, which is only meaningful during training time!"

        self.area_length_per_neighbor = 2*self.MAX_DIST / self.num_neighborhood

        # this assertion ensures the inner area can spawn a neighborhood point cloud
        assert self.MIN_DIST < self.area_length_per_neighbor, "MIN_DIST is too high compared to area_length_per_neighbor! Lower MIN_DIST or lower num_neighborhood instead."
        self.config = config

        self.nuscenes_cache = {}
        self.prepare_nuscenes_ply_colocation()
        print(f"Data size for phase {phase}: {len(self.files)}")

    def prepare_nuscenes_ply_colocation(self):
        # load all frames that have a full spatial neighbourhood
        subset_names = os.listdir(os.path.join(self.root, 'sequences'))
        for dirname in subset_names:
            print(f"Processing log {dirname}")
            fnames = glob.glob(self.root + '/sequences/%s/velodyne/*.bin' % dirname)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_pos = self.get_video_odometry(dirname, return_all=True)
            self.Ts = all_pos[:, :3, 3]

            curr_time = inames[min(int(self.MAX_DIST * 5), int(len(inames)/2))]

            np.random.seed(0)
            while curr_time in inames:
                # find the current neighborhood
                skip, nghb = self._get_neighborhood_frames(curr_time)

                if skip:
                    curr_time += 1
                else:
                    self.files.append((dirname, curr_time, nghb))
                    curr_time += 11 # empirical distance parameter between centers


    def _get_neighborhood_frames(self, frame):
        # list of frame ids belonging to the neighbourhood of the current frame
        list_complement = []
        # indicates that there aren't enough complement frames around this frame
        # so that we should skip this frame
        skip_flag = False
        # Find the frames behind me
        left_frame_bound = max(0, frame-int(10*self.MAX_DIST))
        left_dist = (self.Ts[left_frame_bound:frame] - self.Ts[frame].reshape(1, 3))**2
        left_dist = np.sqrt(left_dist.sum(-1))
        for i in range(int(self.num_neighborhood / 2)):
            area_range_min = max(self.MIN_DIST, self.area_length_per_neighbor*i)
            area_range_max = max(self.MIN_DIST, self.area_length_per_neighbor*(i+1))
            dist_tmp = area_range_min + np.random.rand() * (area_range_max - area_range_min)
            candidates = np.where(left_dist > dist_tmp)[0]
            # print(candidates)
            if len(candidates) == 0:
                # No left-side complement detected
                skip_flag = True
                break
            else:
                list_complement.append(left_frame_bound + candidates[-1])
        
        if skip_flag:
            return (True, [])

        # Find the frames in front of me   
        right_dist = (self.Ts[frame: frame+int(10*self.MAX_DIST)] - self.Ts[frame].reshape(1, 3))**2
        right_dist = np.sqrt(right_dist.sum(-1))
        for i in range(int(self.num_neighborhood / 2)):
            area_range_min = max(self.MIN_DIST, self.area_length_per_neighbor*i)
            area_range_max = max(self.MIN_DIST, self.area_length_per_neighbor*(i+1))
            dist_tmp = area_range_min + np.random.rand() * (area_range_max - area_range_min)
            candidates = np.where(right_dist > dist_tmp)[0]
            if len(candidates) == 0:
                # No right-side complement detected
                skip_flag = True
                list_complement = []
                break
            else:
                list_complement.append(frame + candidates[0])
        return (skip_flag, list_complement)

    def _get_velodyne_fn(self, dirname, t):
        fname = self.root + '/sequences/%s/velodyne/%06d.bin' % (dirname, t)
        return fname

    # simple function for getting the xyz point-cloud w.r.t drive and time
    def _get_xyz(self, drive, time):
        fname = self._get_velodyne_fn(drive, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]

    def __getitem__(self, idx):
        prepare_timer, icp_timer, rot_timer, nn_timer = Timer(), Timer(), Timer(), Timer()
        prepare_timer.tic()
        dirname, t, t_cmpl = self.files[idx]

        positions = self.get_video_odometry(dirname, [t] + t_cmpl)
        pos = positions[0]
        pos_cmpl = positions[1:]

        # load center point cloud
        xyz = self._get_xyz(dirname, t)

        # load neighbourhood point clouds
        xyz_cmpl = []
        for t_tmp in t_cmpl:
            xyz_cmpl.append(self._get_xyz(dirname, t_tmp))
        prepare_timer.toc()

        icp_timer.tic()
        # use world-coordinate label to calculate GT transformation
        def GetListM(pos_core, pos_cmpls):
            return [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(0, int(self.num_neighborhood/2))] + \
                    [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(int(self.num_neighborhood/2), len(pos_cmpls))]
        list_M = GetListM(pos, pos_cmpl)
        icp_timer.toc()
        
        # add random rotation if needed, note that the same rotation is applied to both curr and nghb
        rot_timer.tic()
        if self.random_rotation:
            T0 = sample_random_trans(xyz, self.randg, np.pi / 4)

            xyz = self.apply_transform(xyz, T0)
            for i, xyz_tmp in enumerate(xyz_cmpl):
                Tc = follow_presampled_trans(xyz_tmp, T0)
                xyz_cmpl[i] = self.apply_transform(xyz_tmp, Tc)
                list_M[i] = T0 @ list_M[i] @ np.linalg.inv(Tc)

        # random scaling
        matching_search_voxel_size = self.matching_search_voxel_size
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz = scale * xyz
            for i, xyz_tmp in enumerate(xyz_cmpl):
                xyz_cmpl[i] = scale * xyz_tmp
                list_M[i][:3, 3] = scale * list_M[i][:3, 3]
        rot_timer.toc()

        # voxelization
        nn_timer.tic()
        xyz = torch.from_numpy(xyz)
        for i in range(len(xyz_cmpl)):
            xyz_cmpl[i] = torch.from_numpy(xyz_cmpl[i])
            
        # Make voxelized center points and voxelized center PC
        _, sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
        xyz_th = xyz[sel]
        pcd = make_open3d_point_cloud(xyz[sel])

        # Make both voxelized-unaligned nghb, and PCs
        # They are voxelized with the same index set (sel_nghb) so that both are synonimous w.r.t. point index
        pcd_cmpl = []
        xyz_cmpl_th = []
        for i in range(len(xyz_cmpl)):
            _, sel_nghb = ME.utils.sparse_quantize(xyz_cmpl[i] / self.voxel_size, return_index=True)
            xyz_cmpl_th.append(xyz_cmpl[i][sel_nghb].float())
            pcd_cmpl.append(make_open3d_point_cloud(xyz_cmpl[i][sel_nghb]))
        del sel, sel_nghb

        # Get matches
        group, index, finest_flag, central_distance = get_matching_indices_colocation(pcd, pcd_cmpl, xyz_cmpl_th, list_M, matching_search_voxel_size)
        nn_timer.toc()
        
        group = torch.Tensor(group)
        index = torch.Tensor(index)
        finest_flag = torch.Tensor(finest_flag)
        coords = torch.floor(xyz_th / self.voxel_size).int()
        feats = torch.ones((len(coords), 1)).float()

        if self.calc_distance_err:
            central_distance = torch.Tensor(central_distance)
        else:
            central_distance = torch.Tensor([0])

        coords_cmpl = []
        feats_cmpl = []
        for xyz_tmp_th in xyz_cmpl_th:
            coords_cmpl.append(torch.floor(xyz_tmp_th / self.voxel_size).int())
            feats_cmpl.append(torch.ones((len(coords_cmpl[-1]), 1)).float())

        if self.transform:
            coords, feats = self.transform(coords, feats)
        # print(f"Data loading time: prepare: {prepare_timer.avg}, icp: {icp_timer.avg}, " +
        #       f"rotate & scale: {rot_timer.avg}, nn-search: {nn_timer.avg}")

        return (xyz_th.float(), xyz_cmpl_th, 
                [coords] +  coords_cmpl, [feats] + feats_cmpl, 
                group, index, finest_flag, list_M, central_distance)


from lib.data_loaders import collate_pair_fn
from lib.complement_data_loader import PairComplementKittiDataset, PairComplementNuscenesDataset, collate_complement_pair_fn, collate_debug_pair_fn

ALL_DATASETS = [ColocationKittiDataset, ColocationNuscenesDataset, PairComplementNuscenesDataset, PairComplementKittiDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
    assert phase in ['train', 'trainval', 'val', 'test']
    if shuffle is None:
        shuffle = phase != 'test'

    if config.dataset not in dataset_str_mapping.keys():
        logging.error(f'Dataset {config.dataset}, does not exists in ' +
                       ', '.join(dataset_str_mapping.keys()))

    collate_function = collate_pair_fn

    if phase == 'train' and config.train_dataset in ["ColocationKittiDataset", "ColocationNuscenesDataset", "ColocationKittiSimpleDataset"]:
        collate_function = partial(collate_colocation_fn, config=config)
        Dataset = dataset_str_mapping[config.train_dataset]
    elif config.dataset in ['PairComplementKittiDataset', 'PairComplementNuscenesDataset']:
        if 'debug_need_complement' in [k for (k, v) in config.items()] and config.debug_need_complement:
            collate_function = collate_complement_pair_fn
        else:
            collate_function = collate_debug_pair_fn
        Dataset = dataset_str_mapping[config.dataset]
    else:
        Dataset = dataset_str_mapping[config.dataset]

    use_random_scale = False
    use_random_rotation = False
    transforms = []
    if phase in ['train', 'trainval']:
        use_random_rotation = config.use_random_rotation
        use_random_scale = config.use_random_scale
        transforms += [t.Jitter()]

    dset = Dataset(
        phase,
        transform=t.Compose(transforms),
        random_scale=use_random_scale,
        random_rotation=use_random_rotation,
        manual_seed=True,
        config=config)

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=collate_function,
        pin_memory=False,
        drop_last=True)

    return loader


        




