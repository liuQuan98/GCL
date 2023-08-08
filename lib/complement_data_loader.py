# -*- coding: future_fstrings -*-
#
# Written by Quan Liu <liuquan2017@sjtu.edu.cn>
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

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
import lib.transforms as t
from lib.timer import Timer
from scipy.spatial.transform import Rotation

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


class PairComplementKittiDataset(KittiDataset):
    icp_voxel_size = 0.05 # 0.05 meters, i.e. 5cm

    DATA_FILES = {
        'train': './config/train_kitti.txt',
        'val': './config/val_kitti.txt',
        'test': './config/test_kitti.txt'
    }
    TEST_RANDOM_ROTATION = False
    discard_pairs =[(5, 1151, 1220), (2, 926, 962), (2, 2022, 2054), \
                    (1, 250, 266), (0, 3576, 3609), (2, 2943, 2979), \
                    (1, 411, 423), (2, 2241, 2271), (0, 1536, 1607), \
                    (0, 1338, 1439), (7, 784, 810), (2, 1471, 1498), \
                    (2, 3829, 3862), (0, 1780, 1840), (2, 3294, 3356), \
                    (2, 2420, 2453), (2, 4146, 4206), (0, 2781, 2829), \
                    (0, 3351, 3451), (1, 428, 444), (0, 3073, 3147)]

    
    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PointDataset.__init__(self, phase, transform, random_rotation, 
                              random_scale, manual_seed, config)

        # note: hard-coded
        config.test_augmentation = True
        self.test_augmentation = config.test_augmentation
        self.root = root = config.kitti_root + '/dataset'
        # random_rotation = self.TEST_RANDOM_ROTATION
        self.matching_search_voxel_size = \
            config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.max_correspondence_distance_coarse = self.icp_voxel_size * 15
        self.max_correspondence_distance_fine = self.icp_voxel_size * 1.5
        if "pair_min_dist" in [k for (k, v) in config.items()] and config.pair_min_dist > 0:
            self.MIN_DIST = config.pair_min_dist
        if "pair_max_dist" in [k for (k, v) in config.items()] and config.pair_max_dist > 0 \
            and "pair_min_dist" in [k for (k, v) in config.items()] and config.pair_max_dist >= config.pair_min_dist:
            self.MAX_DIST = config.pair_max_dist

        # pose configuration: use old or new
        try:
            self.use_old_pose = config.use_old_pose
        except:
            self.use_old_pose = True

        if self.use_old_pose:
            self.icp_path = os.path.join(config.kitti_root,'icp')
        else:
            self.icp_path = os.path.join(config.kitti_root,'icp_slam')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        try:
            self.mutate_neighbour = (config.mutate_neighbour_percentage != 0)
            self.mutate_neighbour_percentage = config.mutate_neighbour_percentage
        except:
            self.mutate_neighbour = False
            self.mutate_neighbour_percentage = 0

        logging.info(f"Loading the subset {phase} from {root}")
        self.phase=phase

        self.load_neighbourhood = True
        if self.phase == 'test':
            self.load_neighbourhood = False

        # debug parameters: they are typically set to false.
        self.debug_force_icp_recalculation = False
        if 'debug_force_icp_recalculation' in [k for (k, v) in config.items()]:
            self.debug_force_icp_recalculation = config.debug_force_icp_recalculation
        self.debug_use_old_complement = False
        if 'debug_use_old_complement' in [k for (k, v) in config.items()]:
            self.debug_use_old_complement = config.debug_use_old_complement

        self.min_sample_frame_dist = config.min_sample_frame_dist
        self.complement_pair_dist = config.complement_pair_dist
        self.num_complement_one_side = config.num_complement_one_side
        self.complement_range = self.num_complement_one_side * self.complement_pair_dist
        self.config = config

        if phase == 'test':
            try:
                self.downsample_single = config.downsample_single
            except:
                self.downsample_single = 1.0

        if phase == 'test' and config.LoKITTI == True:
            # load LoKITTI point cloud pairs, instead of generating them based on distance
            self.files = np.load("config/file_LoKITTI_50.npy")
        else:
            self.prepare_kitty_ply(phase)
        print(f"Data size for phase {phase}: {len(self.files)}")

    def prepare_kitty_ply(self, phase):
        # load all frames that have a full spatial neighbourhood
        subset_names = open(self.DATA_FILES[phase]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            print(f"Processing drive {drive_id}")
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            if self.use_old_pose:
                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            else:
                all_pos = self.get_slam_odometry(drive_id, return_all=True)
            self.Ts = all_pos[:, :3, 3]

            curr_time = inames[min(int(self.complement_range * 5), int(len(inames)/2))]

            np.random.seed(0)
            while curr_time in inames:
                # calculate the distance (by random or not)
                dist_tmp = self.MIN_DIST + np.random.rand() * (self.MAX_DIST - self.MIN_DIST)
                right_dist = np.sqrt(((self.Ts[curr_time: curr_time+int(10*self.complement_range)] - 
                                    self.Ts[curr_time].reshape(1, 3))**2).sum(-1))
                # Find the min index
                next_time = np.where(right_dist > dist_tmp)[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                    next_time = next_time[0] + curr_time - 1
                    skip_0, cmpl_0 = self._get_complement_frames(curr_time)
                    skip_1, cmpl_1 = self._get_complement_frames(next_time)
                    skip_2 = (drive_id, curr_time, next_time) in self.discard_pairs
                    if skip_0 or skip_1 or (skip_2 and self.use_old_pose):
                        curr_time += 1
                    else:
                        if self.load_neighbourhood == False:
                            self.files.append((drive_id, curr_time, next_time))
                        else:
                            self.files.append((drive_id, curr_time, next_time, cmpl_0, cmpl_1))
                        curr_time = next_time + 1
                        # curr_time += 8

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

    def _get_complement_frames(self, frame):
        # list of frame ids belonging to the neighbourhood of the current frame
        list_complement = []
        # indicates that there aren't enough complement frames around this frame
        # so that we should skip this frame
        skip_flag = False
        # Find the frames behind me
        left_frame_bound = max(0, frame-int(10*self.complement_range))
        left_dist = (self.Ts[left_frame_bound:frame] - self.Ts[frame].reshape(1, 3))**2
        left_dist = np.sqrt(left_dist.sum(-1))
        for i in range(self.num_complement_one_side):
            dist_tmp = self.complement_pair_dist * (i+1)
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
        right_dist = (self.Ts[frame: frame+int(10*self.complement_range)] - self.Ts[frame].reshape(1, 3))**2
        right_dist = np.sqrt(right_dist.sum(-1))
        for i in range(self.num_complement_one_side):
            dist_tmp = self.complement_pair_dist * (i+1)
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

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    # simple function for getting the xyz point-cloud w.r.t drive and time
    def _get_xyz(self, drive, time):
        fname = self._get_velodyne_fn(drive, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]

    def _get_semantic_label(self, drive, time):
        fname = self._get_labels_fn(drive, time)
        labels = np.fromfile(fname, dtype=np.int16).reshape(-1, 2)
        return labels[:,0]

    # we register next onto current. note tht difference of order here.
    def _get_icp(self, drive, t_curr, t_next, xyz_curr, xyz_next, pos_curr, pos_next):
        key = '%d_%d_%d' % (drive, t_next, t_curr)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in kitti_icp_cache or self.debug_force_icp_recalculation:
            if not os.path.exists(filename) or self.debug_force_icp_recalculation:
                if self.use_old_pose:
                    # work on the downsampled xyzs, 0.05m == 5cm
                    _, sel_curr = ME.utils.sparse_quantize(xyz_curr / self.icp_voxel_size, return_index=True)
                    _, sel0 = ME.utils.sparse_quantize(xyz_next / self.icp_voxel_size, return_index=True)

                    M = (self.velo2cam @ pos_next.T @ np.linalg.inv(pos_curr.T)
                        @ np.linalg.inv(self.velo2cam)).T
                    xyzk_t = self.apply_transform(xyz_next[sel0], M)
                    pcd_curr = make_open3d_point_cloud(xyz_curr[sel_curr])
                    pcd_next = make_open3d_point_cloud(xyzk_t)
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd_next, pcd_curr, 0.2, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                    M2 = M @ reg.transformation
                    # o3d.draw_geometries([pcd0, pcd1])
                else:
                    M2 = np.linalg.inv(pos_curr) @ pos_next
                # write to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]
        return M2
    
    def _get_neighbourhood_icp(self, drive, t_curr, t_cmpls, xyz_curr, xyz_cmpls, pos_curr, pos_cmpls):
        list_M = []
        for i in range(len(t_cmpls)):
            list_M.append(self._get_icp(drive, t_curr, t_cmpls[i], xyz_curr, xyz_cmpls[i], pos_curr, pos_cmpls[i]))
        return list_M

    # registers source onto target (used by multi-way registration)
    def pairwise_registration(self, source, target, pos_source, pos_target):
        # -----------The following code piece is copied from o3d official documentation
        M = (self.velo2cam @ pos_source.T @ np.linalg.inv(pos_target.T)
             @ np.linalg.inv(self.velo2cam)).T
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, 0.2, M,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        transformation_icp = icp_fine.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, self.max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp

    # give the multi-way registration result on one side
    def full_registration(self, pcds, poses):
        # -----------The following code piece is copied from o3d official documentation
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = self.pairwise_registration(
                    pcds[source_id], pcds[target_id], poses[source_id], poses[target_id])
                # print("Build o3d.pipelines.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=True))

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        
        return [pose_graph.nodes[i].pose for i in range(len(pcds))]

    # a piece of socket program between my implementation and o3d official implemnetation
    def multiway_registration(self, drive, t_curr, t_cmpls, xyz_curr, xyz_cmpls, pos_curr, pos_cmpls):
        # check if any of the matrices are not calculated or loaded into cache 
        recalc_flag = False or self.debug_force_icp_recalculation
        reload_flag = False
        for t_next in t_cmpls:
            key = '%d_%d_%d' % (drive, t_next, t_curr)
            filename = self.icp_path + '/' + key + '.npy'
            if key not in kitti_icp_cache:
                if not os.path.exists(filename):
                    recalc_flag = True
                else:
                    reload_flag = True
        
        # if inside cache, then retrieve it
        if not recalc_flag and not reload_flag:
            keys = ['%d_%d_%d' % (drive, t_next, t_curr) for t_next in t_cmpls]
            return [kitti_icp_cache[keys[i]] for i in range(len(keys))]
        
        # if already calculated, but not in cache, then load them
        if not recalc_flag and reload_flag:
            filenames = [self.icp_path + '/%d_%d_%d.npy' % (drive, t_next, t_curr) for t_next in t_cmpls]
            listMs = [np.load(filename) for filename in filenames]
            for i, t_next in enumerate(t_cmpls):
                key = '%d_%d_%d' % (drive, t_next, t_curr)
                kitti_icp_cache[key] = listMs[i]
            return listMs

        # if some of the matrices are not calculated, then re-calculate all of them.
        _, sel = ME.utils.sparse_quantize(xyz_curr / self.icp_voxel_size, return_index=True)
        pcds_left  = [make_open3d_point_cloud(xyz_curr[sel])]
        pcds_right = [make_open3d_point_cloud(xyz_curr[sel])]
        poses_left  = [pos_curr] + pos_cmpls[:self.num_complement_one_side]
        poses_right = [pos_curr] + pos_cmpls[self.num_complement_one_side:]
        for i in range(self.num_complement_one_side):
            _, sel_left = ME.utils.sparse_quantize(xyz_cmpls[i] / self.icp_voxel_size, return_index=True)
            pcds_left.append(make_open3d_point_cloud(xyz_cmpls[i][sel_left]))
            _, sel_right = ME.utils.sparse_quantize(xyz_cmpls[i + self.num_complement_one_side] / self.icp_voxel_size, return_index=True)
            pcds_right.append(make_open3d_point_cloud(xyz_cmpls[i + self.num_complement_one_side][sel_right]))
        
        listM_left = self.full_registration(pcds_left, poses_left)
        listM_right = self.full_registration(pcds_right, poses_right)
        
        listMs = [np.linalg.inv(listM_left[0]) @ listM_left[i] for i in range(1, len(listM_left))] + \
                 [np.linalg.inv(listM_right[0]) @ listM_right[i] for i in range(1, len(listM_right))]
        
        for i, t_next in enumerate(t_cmpls):
            key = '%d_%d_%d' % (drive, t_next, t_curr)
            filename = self.icp_path + '/' + key + '.npy'
            np.save(filename, listMs[i])
            kitti_icp_cache[key] = listMs[i]
        return listMs

    def __getitem__(self, idx):
        # Note that preparation procedures with or without complement frames are very much different,
        # we might as well just throw them in an if-else case, for simplicity of tuning and debugging
        if self.load_neighbourhood:
            prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
            prepare_timer.tic()
            drive, t_0, t_1, t_cmpl_0, t_cmpl_1 = self.files[idx]
            
            if self.use_old_pose:
                all_odometry = self.get_video_odometry(drive, [t_0, t_1] + t_cmpl_0 + t_cmpl_1)
                positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
            else:
                positions = self.get_slam_odometry(drive, [t_0, t_1] + t_cmpl_0 + t_cmpl_1)
            pos_0, pos_1 = positions[0:2]
            pos_cmpl_all = positions[2:]
            pos_cmpl0 = pos_cmpl_all[:2*self.num_complement_one_side]
            pos_cmpl1 = pos_cmpl_all[2*self.num_complement_one_side:]

            if self.mutate_neighbour:
                for pos_cmpl in [pos_cmpl0, pos_cmpl1]:  # two frames: t0 & t1
                    # We denote the position-disturbed heighbourhood frames as 'victims'.
                    num_victims = int(self.mutate_neighbour_percentage*2*self.num_complement_one_side)
                    victim_idxs = np.random.choice(2*self.num_complement_one_side, num_victims, replace=False)
                    for vic_id in victim_idxs:
                        euler_angles=(np.random.rand(3)-0.5)*np.pi*2 # anglez, angley, anglex
                        rot_mutate= Rotation.from_euler('zyx', euler_angles).as_matrix()
                        pos_cmpl[vic_id][:3,:3] = np.dot(pos_cmpl[vic_id][:3,:3], rot_mutate)

            # load two center point clouds
            xyz_0 = self._get_xyz(drive, t_0)
            xyz_1 = self._get_xyz(drive, t_1)

            # load neighbourhood point clouds
            xyz_cmpl_0 = []
            xyz_cmpl_1 = []
            for (t_tmp_0, t_tmp_1) in zip(t_cmpl_0, t_cmpl_1):
                xyz_cmpl_0.append(self._get_xyz(drive, t_tmp_0))
                xyz_cmpl_1.append(self._get_xyz(drive, t_tmp_1))
            prepare_timer.toc()

            icp_timer.tic()
            # use semantic kitti label as GT transformation (acquired using slam)
            if not self.use_old_pose:
                def GetListM(pos_core, pos_cmpls):
                    return [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(0, self.num_complement_one_side)] + \
                           [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(self.num_complement_one_side, len(pos_cmpls))]
                list_M_0 = GetListM(pos_0, pos_cmpl0)
                list_M_1 = GetListM(pos_1, pos_cmpl1)
            # determine and refine rot&trans matrices of the complement frames by icp
            elif 'debug_use_old_complement' in [k for (k, _) in self.config.items()] and self.debug_use_old_complement:
                # -----old method: icp between a cmpl frame and the center frame-----
                list_M_0 = self._get_neighbourhood_icp(drive, t_0, t_cmpl_0, xyz_0, xyz_cmpl_0, pos_0, pos_cmpl0)
                list_M_1 = self._get_neighbourhood_icp(drive, t_1, t_cmpl_1, xyz_1, xyz_cmpl_1, pos_1, pos_cmpl1)
            else:
                # -----new method: o3d official implemnetation of multi-way registration
                list_M_0 = self.multiway_registration(drive, t_0, t_cmpl_0, xyz_0, xyz_cmpl_0, pos_0, pos_cmpl0)
                list_M_1 = self.multiway_registration(drive, t_1, t_cmpl_1, xyz_1, xyz_cmpl_1, pos_1, pos_cmpl1)

            xyz_cmpl_0 = [self.apply_transform(xyz_k, M_k) 
                          for xyz_k, M_k in zip(xyz_cmpl_0, list_M_0)]
            xyz_cmpl_1 = [self.apply_transform(xyz_k, M_k) 
                          for xyz_k, M_k in zip(xyz_cmpl_1, list_M_1)]
            
            # determine icp result between t0 and t1
            # note that we register t0 onto t1, i.e. xyz_1 matches xyz_0 @ M2
            M2 = self._get_icp(drive, t_1, t_0, xyz_1, xyz_0, pos_1, pos_0)
            icp_timer.toc()

            # if idx == 22:
            # print(f"idx: {idx}, drive:{drive}, t0:{t_0}, t1:{t_1}")
            # np.save('pcd0.npy', np.asarray(xyz_0))
            # np.save('pcd1.npy', np.asarray(xyz_1))
            # # np.save('label0.npy', self._get_semantic_label(drive, t_0))
            # # np.save('label1.npy', self._get_semantic_label(drive, t_1))
            # # np.save('trans.npy', np.asarray(M2))
            # print("pcd saved!!!!")
            # raise ValueError
            
            # add random rotation if needed, note that the same rotation is applied to both curr and nghb
            rot_crop_timer.tic()
            if self.random_rotation or self.test_augmentation:
                if self.test_augmentation:
                    T0 = sample_random_trans(xyz_0, self.randg, np.pi*2)
                    T1 = sample_random_trans(xyz_1, self.randg, np.pi*2)
                else:
                    T0 = sample_random_trans(xyz_0, self.randg, np.pi / 4)
                    T1 = sample_random_trans(xyz_1, self.randg, np.pi / 4)
                trans = T1 @ M2 @ np.linalg.inv(T0)

                xyz_0 = self.apply_transform(xyz_0, T0)
                xyz_1 = self.apply_transform(xyz_1, T1)
                xyz_cmpl_0 = [self.apply_transform(xyz_k, T0) 
                              for xyz_k in xyz_cmpl_0]
                xyz_cmpl_1 = [self.apply_transform(xyz_k, T1) 
                              for xyz_k in xyz_cmpl_1]
            else:
                trans = M2

            # if np.linalg.norm(trans[:3,3]) > 20 and idx < 30:
            #     xyz_cmpl_0_bak = xyz_cmpl_0
            #     xyz_cmpl_1_bak = xyz_cmpl_1
            
            # abandon all points that lie out of the scope of the center frame
            # this is because we cannot ask the network to fully imagine
            #   what's there where not even one supporting point exists
            max_dist_square_0 = np.max((xyz_0**2).sum(-1))
            max_dist_square_1 = np.max((xyz_1**2).sum(-1))
            xyz_cmpl_0 = np.concatenate(xyz_cmpl_0, axis=0)
            xyz_cmpl_1 = np.concatenate(xyz_cmpl_1, axis=0)
            xyz_nghb_0 = xyz_cmpl_0[np.where((xyz_cmpl_0**2).sum(-1) < max_dist_square_0)[0]]
            xyz_nghb_1 = xyz_cmpl_1[np.where((xyz_cmpl_1**2).sum(-1) < max_dist_square_1)[0]]
            
            rot_crop_timer.toc()
            del xyz_cmpl_0
            del xyz_cmpl_1

            # apply downsampling on one side during testing
            if self.phase == 'test' and self.downsample_single != 1.0:
                indices = np.random.choice(len(xyz_0), int(len(xyz_0)*self.downsample_single))
                xyz_0 = xyz_0[indices]

            # # if np.linalg.norm(trans[:3,3]) > 20 and idx < 30:
            # pcd_0 = make_open3d_point_cloud(xyz_0)
            # pcd_nghb_0 = make_open3d_point_cloud(xyz_nghb_0)
            # pcd_1 = make_open3d_point_cloud(xyz_1)
            # pcd_nghb_1 = make_open3d_point_cloud(xyz_nghb_1)
            # np.save('pcd0.npy', np.asarray(pcd_0.points))
            # np.save('pcd1.npy', np.asarray(pcd_1.points))
            # np.save('pcd_nghb0.npy', np.asarray(pcd_nghb_0.points))
            # np.save('pcd_nghb1.npy', np.asarray(pcd_nghb_1.points))
            # # np.save('listcmpl_0.npy', xyz_cmpl_0_bak)
            # # np.save('listcmpl_1.npy', xyz_cmpl_1_bak)
            # np.save('trans.npy', np.asarray(trans))
            # print("pcd saved!!!!")
            # raise ValueError

            # random scaling
            matching_search_voxel_size = self.matching_search_voxel_size
            if self.random_scale and random.random() < 0.95:
                scale = self.min_scale + \
                    (self.max_scale - self.min_scale) * random.random()
                matching_search_voxel_size *= scale
                xyz_0 = scale * xyz_0
                xyz_1 = scale * xyz_1
                trans[:3, 3] = scale * trans[:3, 3]

            # voxelization
            xyz_0 = torch.from_numpy(xyz_0)
            xyz_1 = torch.from_numpy(xyz_1)
            xyz_nghb_0 = torch.from_numpy(xyz_nghb_0)
            xyz_nghb_1 = torch.from_numpy(xyz_nghb_1)

            # Make point clouds using voxelized points
            _, sel_0 = ME.utils.sparse_quantize(xyz_0 / self.voxel_size, return_index=True)
            _, sel_1 = ME.utils.sparse_quantize(xyz_1 / self.voxel_size, return_index=True)
            _, sel_nghb_0 = ME.utils.sparse_quantize(xyz_nghb_0 / self.voxel_size, return_index=True)
            _, sel_nghb_1 = ME.utils.sparse_quantize(xyz_nghb_1 / self.voxel_size, return_index=True)
            
            pcd_0 = make_open3d_point_cloud(xyz_0[sel_0])
            pcd_1 = make_open3d_point_cloud(xyz_1[sel_1])

            # Get matches
            matches = get_matching_indices(pcd_0, pcd_1, trans, matching_search_voxel_size)
            if len(matches) < 1000:
                if len(matches) == 0:
                    print("length = 0! Compromising using random relaionships.")
                    print("Remember to remove this pair later!")
                    matches = [(1, 1), (2, 2), (3, 3)]
                # print(f"Matching indices small at {drive}, {t_0}, {t_1},len()={len(matches)}")

            # apply voxelization
            xyz_0_th = xyz_0[sel_0]
            xyz_1_th = xyz_1[sel_1]
            xyz_nghb_0_th = xyz_nghb_0[sel_nghb_0]
            xyz_nghb_1_th = xyz_nghb_1[sel_nghb_1]
            del sel_nghb_0
            del sel_nghb_1
            del sel_0
            del sel_1

            coords_0 = torch.floor(xyz_0_th / self.voxel_size)
            coords_1 = torch.floor(xyz_1_th / self.voxel_size)
            # coords_nghb_0 = torch.floor(xyz_nghb_0_th / self.voxel_size)
            # coords_nghb_1 = torch.floor(xyz_nghb_1_th / self.voxel_size)
            # del xyz_nghb_0_th
            # del xyz_nghb_1_th
            feats_0 = torch.ones((len(coords_0), 1))
            feats_1 = torch.ones((len(coords_1), 1))

            if self.transform:
                coords_0, feats_0 = self.transform(coords_0, feats_0)
                coords_1, feats_1 = self.transform(coords_1, feats_1)
            # print(f"Data loading time: prepare: {prepare_timer.avg}, icp: {icp_timer.avg}, " +
            #       f"rotate & crop: {rot_crop_timer.avg}")

            return (xyz_0_th.float(), xyz_1_th.float(), 
                    xyz_nghb_0_th.float(), xyz_nghb_1_th.float(), 
                    coords_0.int(), coords_1.int(), 
                    feats_0.float(), feats_1.float(), matches, trans)
        else:
            prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
            prepare_timer.tic()
            drive, t_0, t_1 = self.files[idx]
            if self.use_old_pose:
                all_odometry = self.get_video_odometry(drive, [t_0, t_1])
                positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
            else:
                positions = self.get_slam_odometry(drive, [t_0, t_1])
            pos_0, pos_1 = positions[0:2]

            # load two center point clouds
            xyz_0 = self._get_xyz(drive, t_0)
            xyz_1 = self._get_xyz(drive, t_1)

            _, sel_curr_0 = ME.utils.sparse_quantize(xyz_0 / 0.05, return_index=True)
            _, sel_curr_1 = ME.utils.sparse_quantize(xyz_1 / 0.05, return_index=True)
            pcd_0 = make_open3d_point_cloud(xyz_0[sel_curr_0])
            pcd_1 = make_open3d_point_cloud(xyz_1[sel_curr_1])
            del sel_curr_0
            del sel_curr_1
            prepare_timer.toc()

            icp_timer.tic()
            # determine icp result between t0 and t1
            # note that we register t0 onto t1, i.e. pcd1 matches pcd0 @ M2
            M2 = self._get_icp(drive, t_1, t_0, xyz_1, xyz_0, pos_1, pos_0)
            icp_timer.toc()

            # apply downsampling on one side during testing
            if self.phase == 'test' and self.downsample_single != 1.0:
                indices = np.random.choice(len(xyz_0), int(len(xyz_0)*self.downsample_single))
                xyz_0 = xyz_0[indices]
            
            # add random rotation if needed, note that the same rotation is applied to both curr and nghb
            rot_crop_timer.tic()
            if self.random_rotation:
                T0 = sample_random_trans(xyz_0, self.randg, np.pi / 4)
                T1 = sample_random_trans(xyz_1, self.randg, np.pi / 4)
                trans = T1 @ M2 @ np.linalg.inv(T0)

                xyz_0 = self.apply_transform(xyz_0, T0)
                xyz_1 = self.apply_transform(xyz_1, T1)
            else:
                trans = M2
            rot_crop_timer.toc()

            # print(f"idx: {idx}, drive:{drive}, t0:{t_0}, t1:{t_1}")
            # np.save('pcd0.npy', np.asarray(xyz_0))
            # np.save('pcd1.npy', np.asarray(xyz_1))
            # # np.save('label0.npy', self._get_semantic_label(drive, t_0))
            # # np.save('label1.npy', self._get_semantic_label(drive, t_1))
            # # np.save('trans.npy', np.asarray(M2))
            # print("pcd saved!!!!")
            # raise ValueError

            # random scaling
            matching_search_voxel_size = self.matching_search_voxel_size
            if self.random_scale and random.random() < 0.95:
                scale = self.min_scale + \
                    (self.max_scale - self.min_scale) * random.random()
                matching_search_voxel_size *= scale
                xyz_0 = scale * xyz_0
                xyz_1 = scale * xyz_1
                trans[:3, 3] = scale * trans[:3, 3]

            # voxelization
            xyz_0 = torch.from_numpy(xyz_0)
            xyz_1 = torch.from_numpy(xyz_1)

            # Make point clouds using voxelized points
            _, sel_0 = ME.utils.sparse_quantize(xyz_0 / self.voxel_size, return_index=True)
            _, sel_1 = ME.utils.sparse_quantize(xyz_1 / self.voxel_size, return_index=True)
            
            pcd_0 = make_open3d_point_cloud(xyz_0[sel_0])
            pcd_1 = make_open3d_point_cloud(xyz_1[sel_1])

            # Get matches
            matches = get_matching_indices(pcd_0, pcd_1, trans, matching_search_voxel_size)
            if len(matches) < 1000:
                if len(matches) == 0:
                    print("length = 0! Compromising using random relaionships.")
                    print("Remember to remove this pair later!")
                    matches = [(1, 1), (2, 2), (3, 3)]
                # print(f"Matching indices small at {drive}, {t_0}, {t_1},len()={len(matches)}")

            # apply voxelization
            xyz_0_th = xyz_0[sel_0]
            xyz_1_th = xyz_1[sel_1]
            del sel_0
            del sel_1

            coords_0 = torch.floor(xyz_0_th / self.voxel_size)
            coords_1 = torch.floor(xyz_1_th / self.voxel_size)
            feats_0 = torch.ones((len(coords_0), 1))
            feats_1 = torch.ones((len(coords_1), 1))

            if self.transform:
                coords_0, feats_0 = self.transform(coords_0, feats_0)
                coords_1, feats_1 = self.transform(coords_1, feats_1)

            # note: we now provide voxelized neighbourhood.
            # whether unvoxelized pcd performs better is still unclear.
            return (xyz_0_th.float(), xyz_1_th.float(), 
                    coords_0.int(), coords_1.int(), 
                    feats_0.float(), feats_1.float(), matches, trans)


class PairComplementNuscenesDataset(KittiDataset):
    icp_voxel_size = 0.05 # 0.05 meters, i.e. 5cm
    TEST_RANDOM_ROTATION = False
    
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
        random_rotation = self.TEST_RANDOM_ROTATION
        self.matching_search_voxel_size = \
            config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.max_correspondence_distance_coarse = self.icp_voxel_size * 15
        self.max_correspondence_distance_fine = self.icp_voxel_size * 1.5
        if "pair_min_dist" in [k for (k, v) in config.items()] and config.pair_min_dist > 0:
            self.MIN_DIST = config.pair_min_dist
        if "pair_max_dist" in [k for (k, v) in config.items()] and config.pair_max_dist > 0 \
            and "pair_min_dist" in [k for (k, v) in config.items()] and config.pair_max_dist >= config.pair_min_dist:
            self.MAX_DIST = config.pair_max_dist

        self.phase = phase
        if phase == 'test':
            try:
                self.downsample_single = config.downsample_single
            except:
                self.downsample_single = 1.0

        try:
            self.mutate_neighbour = (config.mutate_neighbour_percentage != 0)
            self.mutate_neighbour_percentage = config.mutate_neighbour_percentage
        except:
            self.mutate_neighbour = False
            self.mutate_neighbour_percentage = 0

        # pose configuration: use old or new
        try:
            self.use_old_pose = config.use_old_pose
        except:
            self.use_old_pose = True
        assert config.use_old_pose is True, "no slam-based position available!"

        self.icp_path = os.path.join(config.kitti_root,'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        self.load_neighbourhood = True
        if self.phase == 'test':
            self.load_neighbourhood = False

        self.min_sample_frame_dist = config.min_sample_frame_dist
        self.complement_pair_dist = config.complement_pair_dist
        self.num_complement_one_side = config.num_complement_one_side
        self.complement_range = self.num_complement_one_side * self.complement_pair_dist
        self.config = config

        self.files = []
        self.nuscenes_icp_cache = {}
        self.nuscenes_cache = {}

        # load LoNuscenes point cloud pairs, instead of generating them based on distance
        if phase == 'test' and config.LoNUSCENES == True:
            self.files = np.load("config/file_LoNUSCENES_50.npy", allow_pickle=True)
        else:
            logging.info(f"Loading the subset {phase} from {root}")

            # load all frames that have a full spatial neighbourhood
            subset_names = os.listdir(os.path.join(self.root, 'sequences'))
            for dirname in subset_names:
                print(f"Processing log {dirname}")
                fnames = glob.glob(self.root + '/sequences/%s/velodyne/*.bin' % dirname)
                assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

                all_pos = self.get_video_odometry(dirname, return_all=True)
                self.Ts = all_pos[:, :3, 3]

                curr_time = inames[min(int(self.complement_range * 5), int(len(inames)/2))]

                np.random.seed(0)
                while curr_time in inames:
                    # calculate the distance (by random or not)
                    dist_tmp = self.MIN_DIST + np.random.rand() * (self.MAX_DIST - self.MIN_DIST)

                    right_dist = np.sqrt(((self.Ts[curr_time: curr_time+int(10*self.complement_range)] - 
                                        self.Ts[curr_time].reshape(1, 3))**2).sum(-1))
                    # Find the min index
                    next_time = np.where(right_dist > dist_tmp)[0]
                    if len(next_time) == 0:
                        curr_time += 1
                    else:
                        # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/nuscenes/process_nuscenes_data.m#L44
                        next_time = next_time[0] + curr_time - 1
                        skip_0, cmpl_0 = self._get_complement_frames(curr_time)
                        skip_1, cmpl_1 = self._get_complement_frames(next_time)
                        if skip_0 or skip_1:
                            curr_time += 1
                        else:
                            self.files.append((dirname, curr_time, next_time, cmpl_0, cmpl_1))
                            curr_time = next_time + 1
        if phase == 'train':
            self.files = self.files[::3]
            self.files = self.files[:1200]
        print(f"Data size for phase {phase}: {len(self.files)}")
    
    def _get_complement_frames(self, frame):
        # list of frame ids belonging to the neighbourhood of the current frame
        list_complement = []
        # indicates that there aren't enough complement frames around this frame
        # so that we should skip this frame
        skip_flag = False
        # Find the frames behind me
        left_frame_bound = max(0, frame-int(10*self.complement_range))
        left_dist = (self.Ts[left_frame_bound:frame] - self.Ts[frame].reshape(1, 3))**2
        left_dist = np.sqrt(left_dist.sum(-1))
        for i in range(self.num_complement_one_side):
            dist_tmp = self.complement_pair_dist * (i+1)
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
        right_dist = (self.Ts[frame: frame+int(10*self.complement_range)] - self.Ts[frame].reshape(1, 3))**2
        right_dist = np.sqrt(right_dist.sum(-1))
        for i in range(self.num_complement_one_side):
            dist_tmp = self.complement_pair_dist * (i+1)
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
        # Note that preparation procedures with or without complement frames are very much different,
        # we might as well just throw them in an if-else case, for simplicity of tuning and debugging
        if self.load_neighbourhood:
            prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
            prepare_timer.tic()
            dirname, t_0, t_1, t_cmpl_0, t_cmpl_1 = self.files[idx]

            positions = self.get_video_odometry(dirname, [t_0, t_1] + t_cmpl_0 + t_cmpl_1)

            pos_0, pos_1 = positions[0:2]
            pos_cmpl_all = positions[2:]
            pos_cmpl0 = pos_cmpl_all[:2*self.num_complement_one_side]
            pos_cmpl1 = pos_cmpl_all[2*self.num_complement_one_side:]

            if self.mutate_neighbour:
                for pos_cmpl in [pos_cmpl0, pos_cmpl1]:  # two frames: t0 & t1
                    # We denote the position-disturbed heighbourhood frames as 'victims'.
                    num_victims = int(self.mutate_neighbour_percentage*2*self.num_complement_one_side)
                    victim_idxs = np.random.choice(2*self.num_complement_one_side, num_victims, replace=False)
                    for vic_id in victim_idxs:
                        euler_angles=(np.random.rand(3)-0.5)*np.pi*2 # anglez, angley, anglex
                        rot_mutate= Rotation.from_euler('zyx', euler_angles).as_matrix()
                        pos_cmpl[vic_id][:3,:3] = np.dot(pos_cmpl[vic_id][:3,:3], rot_mutate)

            # load two center point clouds
            xyz_0 = self._get_xyz(dirname, t_0)
            xyz_1 = self._get_xyz(dirname, t_1)

            # load neighbourhood point clouds
            xyz_cmpl_0 = []
            xyz_cmpl_1 = []
            for (t_tmp_0, t_tmp_1) in zip(t_cmpl_0, t_cmpl_1):
                xyz_cmpl_0.append(self._get_xyz(dirname, t_tmp_0))
                xyz_cmpl_1.append(self._get_xyz(dirname, t_tmp_1))
            prepare_timer.toc()

            icp_timer.tic()
            # use semantic kitti label as GT transformation (acquired using slam)
            def GetListM(pos_core, pos_cmpls):
                return [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(0, self.num_complement_one_side)] + \
                        [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(self.num_complement_one_side, len(pos_cmpls))]
            list_M_0 = GetListM(pos_0, pos_cmpl0)
            list_M_1 = GetListM(pos_1, pos_cmpl1)

            xyz_cmpl_0 = [self.apply_transform(xyz_k, M_k) 
                          for xyz_k, M_k in zip(xyz_cmpl_0, list_M_0)]
            xyz_cmpl_1 = [self.apply_transform(xyz_k, M_k) 
                          for xyz_k, M_k in zip(xyz_cmpl_1, list_M_1)]
            
            # determine icp result between t0 and t1
            # note that we register t0 onto t1, i.e. xyz_1 matches xyz_0 @ M2
            M2 = np.linalg.inv(positions[1]) @ positions[0]
            icp_timer.toc()
            
            # add random rotation if needed, note that the same rotation is applied to both curr and nghb
            rot_crop_timer.tic()
            if self.random_rotation:
                T0 = sample_random_trans(xyz_0, self.randg, np.pi / 4)
                T1 = sample_random_trans(xyz_1, self.randg, np.pi / 4)
                trans = T1 @ M2 @ np.linalg.inv(T0)

                xyz_0 = self.apply_transform(xyz_0, T0)
                xyz_1 = self.apply_transform(xyz_1, T1)
                xyz_cmpl_0 = [self.apply_transform(xyz_k, T0) 
                              for xyz_k in xyz_cmpl_0]
                xyz_cmpl_1 = [self.apply_transform(xyz_k, T1) 
                              for xyz_k in xyz_cmpl_1]
            else:
                trans = M2
            
            # abandon all points that lie out of the scope of the center frame
            # this is because we cannot ask the network to fully imagine
            #   what's there where not even one supporting point exists
            max_dist_square_0 = np.max((xyz_0**2).sum(-1))
            max_dist_square_1 = np.max((xyz_1**2).sum(-1))
            xyz_cmpl_0 = np.concatenate(xyz_cmpl_0, axis=0)
            xyz_cmpl_1 = np.concatenate(xyz_cmpl_1, axis=0)
            xyz_nghb_0 = xyz_cmpl_0[np.where((xyz_cmpl_0**2).sum(-1) < max_dist_square_0)[0]]
            xyz_nghb_1 = xyz_cmpl_1[np.where((xyz_cmpl_1**2).sum(-1) < max_dist_square_1)[0]]
            
            rot_crop_timer.toc()
            del xyz_cmpl_0
            del xyz_cmpl_1

            # apply downsampling on one side during testing
            if self.phase == 'test' and self.downsample_single != 1.0:
                indices = np.random.choice(len(xyz_0), int(len(xyz_0)*self.downsample_single))
                xyz_0 = xyz_0[indices]

            # random scaling
            matching_search_voxel_size = self.matching_search_voxel_size
            if self.random_scale and random.random() < 0.95:
                scale = self.min_scale + \
                    (self.max_scale - self.min_scale) * random.random()
                matching_search_voxel_size *= scale
                xyz_0 = scale * xyz_0
                xyz_1 = scale * xyz_1

            # voxelization
            xyz_0 = torch.from_numpy(xyz_0)
            xyz_1 = torch.from_numpy(xyz_1)
            xyz_nghb_0 = torch.from_numpy(xyz_nghb_0)
            xyz_nghb_1 = torch.from_numpy(xyz_nghb_1)

            # Make point clouds using voxelized points
            _, sel_0 = ME.utils.sparse_quantize(xyz_0 / self.voxel_size, return_index=True)
            _, sel_1 = ME.utils.sparse_quantize(xyz_1 / self.voxel_size, return_index=True)
            _, sel_nghb_0 = ME.utils.sparse_quantize(xyz_nghb_0 / self.voxel_size, return_index=True)
            _, sel_nghb_1 = ME.utils.sparse_quantize(xyz_nghb_1 / self.voxel_size, return_index=True)
            
            pcd_0 = make_open3d_point_cloud(xyz_0[sel_0])
            pcd_1 = make_open3d_point_cloud(xyz_1[sel_1])

            # Get matches
            matches = get_matching_indices(pcd_0, pcd_1, trans, matching_search_voxel_size)
            if len(matches) < 1000:
                if len(matches) == 0:
                    print("length = 0! Compromising using random relaionships.")
                    print("Remember to remove this pair later!")
                    matches = [(1, 1), (2, 2), (3, 3)]
                # print(f"Matching indices small at {drive}, {t_0}, {t_1},len()={len(matches)}")

            # apply voxelization
            xyz_0_th = xyz_0[sel_0]
            xyz_1_th = xyz_1[sel_1]
            xyz_nghb_0_th = xyz_nghb_0[sel_nghb_0]
            xyz_nghb_1_th = xyz_nghb_1[sel_nghb_1]
            del sel_nghb_0
            del sel_nghb_1
            del sel_0
            del sel_1

            coords_0 = torch.floor(xyz_0_th / self.voxel_size)
            coords_1 = torch.floor(xyz_1_th / self.voxel_size)
            feats_0 = torch.ones((len(coords_0), 1))
            feats_1 = torch.ones((len(coords_1), 1))

            if self.transform:
                coords_0, feats_0 = self.transform(coords_0, feats_0)
                coords_1, feats_1 = self.transform(coords_1, feats_1)
            # print(f"Data loading time: prepare: {prepare_timer.avg}, icp: {icp_timer.avg}, " +
            #       f"rotate & crop: {rot_crop_timer.avg}")

            return (xyz_0_th.float(), xyz_1_th.float(), 
                    xyz_nghb_0_th.float(), xyz_nghb_1_th.float(), 
                    coords_0.int(), coords_1.int(), 
                    feats_0.float(), feats_1.float(), matches, trans)

        else:
            prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
            prepare_timer.tic()
            try:
                dirname, t_0, t_1, t_cmpl_0, t_cmpl_1 = self.files[idx]
            except:
                dirname, t_0, t_1 = self.files[idx]
            # print(self.files[idx])
            positions = self.get_video_odometry(dirname, [t_0, t_1])

            pos_0, pos_1 = positions[0:2]

            # load two center point clouds
            xyz_0 = self._get_xyz(dirname, t_0)
            xyz_1 = self._get_xyz(dirname, t_1)
            prepare_timer.toc()

            icp_timer.tic()
            # determine icp result between t0 and t1
            # note that we register t0 onto t1, i.e. pcd1 matches pcd0 @ M2
            M2 = np.linalg.inv(positions[1]) @ positions[0]
            icp_timer.toc()
            
            # add random rotation if needed, note that the same rotation is applied to both curr and nghb
            rot_crop_timer.tic()
            if self.random_rotation:
                T0 = sample_random_trans(xyz_0, self.randg, np.pi / 4)
                T1 = sample_random_trans(xyz_1, self.randg, np.pi / 4)
                trans = T1 @ M2 @ np.linalg.inv(T0)

                xyz_0 = self.apply_transform(xyz_0, T0)
                xyz_1 = self.apply_transform(xyz_1, T1)
            else:
                trans = M2
            rot_crop_timer.toc()

            # apply downsampling on one side during testing
            if self.phase == 'test' and self.downsample_single != 1.0:
                indices = np.random.choice(len(xyz_0), int(len(xyz_0)*self.downsample_single))
                xyz_0 = xyz_0[indices]

            # random scaling
            matching_search_voxel_size = self.matching_search_voxel_size
            if self.random_scale and random.random() < 0.95:
                scale = self.min_scale + \
                    (self.max_scale - self.min_scale) * random.random()
                matching_search_voxel_size *= scale
                xyz_0 = scale * xyz_0
                xyz_1 = scale * xyz_1

            # voxelization
            xyz_0 = torch.from_numpy(xyz_0)
            xyz_1 = torch.from_numpy(xyz_1)

            # Make point clouds using voxelized points
            _, sel_0 = ME.utils.sparse_quantize(xyz_0 / self.voxel_size, return_index=True)
            _, sel_1 = ME.utils.sparse_quantize(xyz_1 / self.voxel_size, return_index=True)
            
            pcd_0 = make_open3d_point_cloud(xyz_0[sel_0])
            pcd_1 = make_open3d_point_cloud(xyz_1[sel_1])

            # Get matches
            matches = get_matching_indices(pcd_0, pcd_1, trans, matching_search_voxel_size)
            if len(matches) < 1000:
                if len(matches) == 0:
                    print("length = 0! Compromising using random relaionships.")
                    print("Remember to remove this pair later!")
                    matches = [(1, 1), (2, 2), (3, 3)]
                # print(f"Matching indices small at {drive}, {t_0}, {t_1},len()={len(matches)}")

            # apply voxelization
            xyz_0_th = xyz_0[sel_0]
            xyz_1_th = xyz_1[sel_1]
            del sel_0
            del sel_1

            coords_0 = torch.floor(xyz_0_th / self.voxel_size)
            coords_1 = torch.floor(xyz_1_th / self.voxel_size)
            feats_0 = torch.ones((len(coords_0), 1))
            feats_1 = torch.ones((len(coords_1), 1))

            if self.transform:
                coords_0, feats_0 = self.transform(coords_0, feats_0)
                coords_1, feats_1 = self.transform(coords_1, feats_1)

            # note: we now provide voxelized neighbourhood.
            # whether unvoxelized pcd performs better is still unclear.
            return (xyz_0_th.float(), xyz_1_th.float(), 
                    coords_0.int(), coords_1.int(), 
                    feats_0.float(), feats_1.float(), matches, trans)


def collate_complement_pair_fn(list_data):
    xyz0, xyz1, xyz_nghb0, xyz_nghb1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
        zip(*list_data))
    # xyz_batch0, xyz_batch1 = [], []
    # xyz_nghb_batch0, xyz_nghb_batch1 = [], []
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]
        if len(matching_inds[batch_id]) != 0:
            trans_batch.append(to_tensor(trans[batch_id]))
            matching_inds_batch.append(
                torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
            len_batch.append([N0, N1])
        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    # Note that outputs here obey different data format.
    # All items starting with 'pcd' are lists of Tensors, for example:
    #  results['pcd0'][0]   # this is a Tensor
    #  results['pcd0']      # this is a list of Tensors
    # while all others are already concatenated and do not need '[i]' to access:
    #  results['T_gt']      # this is already a 4*4 rot&trans matrix (Tensor)
    return {
        'pcd0': xyz0,
        'pcd1': xyz1,
        'pcd_nghb0': xyz_nghb0,
        'pcd_nghb1': xyz_nghb1,
        'sinput0_C': coords_batch0,
        'sinput0_F': feats_batch0.float(),
        'sinput1_C': coords_batch1,
        'sinput1_F': feats_batch1.float(),
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch
    }


def collate_debug_pair_fn(list_data):
    xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
        zip(*list_data))
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]
        if len(matching_inds[batch_id]) != 0:
            trans_batch.append(to_tensor(trans[batch_id]))
            matching_inds_batch.append(
                torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
            len_batch.append([N0, N1])
        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    # Note that outputs here obey different data format.
    # All items starting with 'pcd' are lists of Tensors, for example:
    #  results['pcd0'][0]   # this is a Tensor
    #  results['pcd0']      # this is a list of Tensors
    # while all others are already concatenated and do not need '[i]' to access:
    #  results['T_gt']      # this is already a 4*4 rot&trans matrix (Tensor)
    return {
        'pcd0': xyz0,
        'pcd1': xyz1,
        'sinput0_C': coords_batch0,
        'sinput0_F': feats_batch0.float(),
        'sinput1_C': coords_batch1,
        'sinput1_F': feats_batch1.float(),
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch
    }


from lib.data_loaders import ThreeDMatchPairDataset, KITTIPairDataset, KITTINMPairDataset, KITTIRandDistPairDataset, collate_pair_fn
ALL_DATASETS = [ThreeDMatchPairDataset, KITTIPairDataset, KITTINMPairDataset, KITTIRandDistPairDataset, PairComplementKittiDataset, PairComplementNuscenesDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
    assert phase in ['train', 'trainval', 'val', 'test']
    if shuffle is None:
        shuffle = phase != 'test'

    if config.dataset not in dataset_str_mapping.keys():
        logging.error(f'Dataset {config.dataset}, does not exists in ' +
                       ', '.join(dataset_str_mapping.keys()))

    collate_function = collate_pair_fn
    if config.dataset in ['PairComplementKittiDataset', 'PairComplementNuscenesDataset']:
        if phase == 'test':
            collate_function = collate_debug_pair_fn
        else:
            collate_function = collate_complement_pair_fn

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

