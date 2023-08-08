import sys

sys.path.append('../')
import open3d
import numpy as np
import time
import os
# from ThreeDMatch.Test.tools import get_pcd, get_ETH_keypts, get_desc, loadlog
from sklearn.neighbors import KDTree
import glob

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from model import load_model
from lib.eval import find_nn_gpu
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
import MinkowskiEngine as ME

from pytorch3d.ops.knn import knn_points
import pytorch3d

def get_pcd(pcdpath, filename):
    return open3d.io.read_point_cloud(os.path.join(pcdpath, filename + '.ply'))

def get_ETH_keypts(pcd, keyptspath, filename):
    pts = np.array(pcd.points)
    key_ind = np.loadtxt(os.path.join(keyptspath, filename + '_Keypoints.txt'), dtype=np.int32)
    keypts = pts[key_ind]
    return keypts

def get_desc(descpath, filename, desc_name):
    if desc_name == '3dmatch':
        desc = np.fromfile(os.path.join(descpath, filename + '.desc.3dmatch.bin'), dtype=np.float32)
        num_desc = int(desc[0])
        desc_size = int(desc[1])
        desc = desc[2:].reshape([num_desc, desc_size])
    elif desc_name == 'SpinNet':
        desc = np.load(os.path.join(descpath, filename + '.desc.SpinNet.bin.npy'))
    else:
        print("No such descriptor")
        exit(-1)
    return desc

def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result

def calculate_M(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 512]
    """

    kdtree_s = KDTree(target_desc)
    sourceNNdis, sourceNNidx = kdtree_s.query(source_desc, 1)
    kdtree_t = KDTree(source_desc)
    targetNNdis, targetNNidx = kdtree_t.query(target_desc, 1)
    result = []
    for i in range(len(sourceNNidx)):
        if targetNNidx[sourceNNidx[i]] == i:
            result.append([i, sourceNNidx[i][0]])
    return np.array(result)


def prepare_pcd_to_input(xyz_0, xyz_1):
    # voxelization
    xyz_0 = torch.from_numpy(xyz_0)
    xyz_1 = torch.from_numpy(xyz_1)

    _, sel_0 = ME.utils.sparse_quantize(xyz_0 / voxel_size, return_index=True)
    _, sel_1 = ME.utils.sparse_quantize(xyz_1 / voxel_size, return_index=True)

    # apply voxelization
    xyz_0_th = xyz_0[sel_0]
    xyz_1_th = xyz_1[sel_1]
    del sel_0
    del sel_1

    coords_0 = torch.floor(xyz_0_th / voxel_size)
    coords_1 = torch.floor(xyz_1_th /  voxel_size)
    feats_0 = torch.ones((len(coords_0), 1))
    feats_1 = torch.ones((len(coords_1), 1))

    coords_batch0, feats_batch0 = ME.utils.sparse_collate([coords_0], [feats_0])
    coords_batch1, feats_batch1 = ME.utils.sparse_collate([coords_1], [feats_1])

    sinput0 = ME.SparseTensor(
        feats_batch0.to(device), coordinates=coords_batch0.to(device))
    sinput1 = ME.SparseTensor(
        feats_batch1.to(device), coordinates=coords_batch1.to(device))
    
    return sinput0, sinput1, xyz_0_th, xyz_1_th


def find_nearest_voxel_feature(full, partial, features):
    P1 = pytorch3d.structures.Pointclouds([partial])
    P2 = pytorch3d.structures.Pointclouds([full])

    P1_F = P1.points_padded()
    P2_F = P2.points_padded()
    P1_N = P1.num_points_per_cloud()
    P2_N = P2.num_points_per_cloud()

    _, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=1)
    idx_1 = idx_1[:, :, 0][0]

    return features[idx_1]


def register2Fragments(id1, id2, keyptspath, resultpath, desc_name='ppf'):
    cloud_bin_s = f'Hokuyo_{id1}'
    cloud_bin_t = f'Hokuyo_{id2}'
    write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
    if os.path.exists(os.path.join(resultpath, write_file)):
        #      print(f"{write_file} already exists.")
        return 0, 0, 0
    pcd_s = get_pcd(pcdpath, cloud_bin_s)
    source_keypts = get_ETH_keypts(pcd_s, keyptspath, cloud_bin_s)
    pcd_t = get_pcd(pcdpath, cloud_bin_t)
    target_keypts = get_ETH_keypts(pcd_t, keyptspath, cloud_bin_t)
    # print(source_keypts.shape)

    sinput0, sinput1, voxel0, voxel1 = prepare_pcd_to_input(np.array(pcd_s.points), np.array(pcd_t.points))

    enc0 = model(sinput0)
    F0 = enc0.F.detach()
    enc1 = model(sinput1)
    F1 = enc1.F.detach()
    source_desc = find_nearest_voxel_feature(voxel0.to(device), torch.from_numpy(source_keypts).to(device), F0).cpu()
    target_desc = find_nearest_voxel_feature(voxel1.to(device), torch.from_numpy(target_keypts).to(device), F1).cpu()
    # source_desc = get_desc(descpath, cloud_bin_s, desc_name=desc_name)
    # target_desc = get_desc(descpath, cloud_bin_t, desc_name=desc_name)
    # source_desc = np.nan_to_num(source_desc)
    # target_desc = np.nan_to_num(target_desc)

    key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
    if key not in gtLog.keys():
        num_inliers = 0
        inlier_ratio = 0
        gt_flag = 0
    else:
        # find mutually cloest point.
        corr = calculate_M(source_desc, target_desc)

        gtTrans = gtLog[key]
        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = open3d.geometry.PointCloud()
        frag2_pc.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gtTrans)
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < 0.1)
        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1

        # calculate the transformation matrix using RANSAC, this is for Registration Recall.
        source_pcd = open3d.geometry.PointCloud()
        source_pcd.points = open3d.utility.Vector3dVector(source_keypts)
        target_pcd = open3d.geometry.PointCloud()
        target_pcd.points = open3d.utility.Vector3dVector(target_keypts)
        s_desc = open3d.pipelines.registration.Feature()
        s_desc.data = source_desc.T
        t_desc = open3d.pipelines.registration.Feature()
        t_desc.data = target_desc.T
        result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, s_desc, t_desc, False,
            0.05,
            open3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
            open3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
        # write the transformation matrix into .log file for evaluation.
        with open(os.path.join(logpath, f'{desc_name}_{timestr}.log'), 'a+') as f:
            trans = result.transformation
            trans = np.linalg.inv(trans)
            s1 = f'{id1}\t {id2}\t  37\n'
            f.write(s1)
            f.write(f"{trans[0, 0]}\t {trans[0, 1]}\t {trans[0, 2]}\t {trans[0, 3]}\t \n")
            f.write(f"{trans[1, 0]}\t {trans[1, 1]}\t {trans[1, 2]}\t {trans[1, 3]}\t \n")
            f.write(f"{trans[2, 0]}\t {trans[2, 1]}\t {trans[2, 2]}\t {trans[2, 3]}\t \n")
            f.write(f"{trans[3, 0]}\t {trans[3, 1]}\t {trans[3, 2]}\t {trans[3, 3]}\t \n")

    s = f"{cloud_bin_s}\t{cloud_bin_t}\t{num_inliers}\t{inlier_ratio:.8f}\t{gt_flag}"
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'w+') as f:
        f.write(s)
    return num_inliers, inlier_ratio, gt_flag


def read_register_result(id1, id2):
    cloud_bin_s = f'Hokuyo_{id1}'
    cloud_bin_t = f'Hokuyo_{id2}'
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
        content = f.readlines()
    nums = content[0].replace("\n", "").split("\t")[2:5]
    return nums


if __name__ == '__main__':
    scene_list = [
        'gazebo_summer',
        'gazebo_winter',
        'wood_autmn',
        'wood_summer',
    ]
    timestr = sys.argv[1]
    inliers_list = []
    recall_list = []

    # parameters
    voxel_size = 0.05

    Model = load_model("ResUNetFatBN")
    model = Model(
        1,
        32,
        bn_momentum=0.05,
        conv1_kernel_size=5,
        normalize_feature=True)
    desc_name = 'GCL+Conv'
    checkpoint = torch.load('./outputs/Experiments/PairComplementKittiDataset-v0.3/FinestContrastiveLossTrainer/ResUNetFatBN/SGD-lr1e-1-e200-b4i1-modelnout32/2023-02-05_21-25-40/best_val_checkpoint.pth')
    # desc_name = 'FCGF_50'
    # checkpoint = torch.load('./outputs/Experiments/FCGF_pretrained/2023-02-13_19-48-37_FCGF_5_20/best_val_checkpoint.pth')
    # checkpoint = torch.load('./outputs/Experiments/FCGF_pretrained/2023-02-14_10-31-44_FCGF_5_50/best_val_checkpoint.pth')

    model.load_state_dict(checkpoint['state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    for scene in scene_list:
        pcdpath = f"./ETH/{scene}/"
        interpath = f"./ETH/{scene}/01_Keypoints/"
        gtpath = f'./ETH/{scene}/'
        keyptspath = interpath  # os.path.join(interpath, "keypoints/")
        logpath = f"./ETH/log_result/{scene}-evaluation"
        gtLog = loadlog(gtpath)
        resultpath = os.path.join("./ETH", f"pred_result/{scene}/{desc_name}_result_{timestr}")
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        if not os.path.exists(logpath):
            os.makedirs(logpath)

        # register each pair
        fragments = glob.glob(pcdpath + '*.ply')
        num_frag = len(fragments)
        print(f"Start Evaluating Scene {scene}")
        start_time = time.time()
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                num_inliers, inlier_ratio, gt_flag = register2Fragments(id1, id2, keyptspath, resultpath,
                                                                        desc_name)
        print(f"Finish Evaluation, time: {time.time() - start_time:.2f}s")

        # evaluate
        result = []
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                line = read_register_result(id1, id2)
                result.append([int(line[0]), float(line[1]), int(line[2])])
        result = np.array(result)
        indices_results = np.sum(result[:, 2] == 1)
        correct_match = np.sum(result[:, 1] > 0.05)
        recall = float(correct_match / indices_results) * 100
        print(f"Correct Match {correct_match}, ground truth Match {indices_results}")
        print(f"Recall {recall}%")
        ave_num_inliers = np.sum(np.where(result[:, 1] > 0.05, result[:, 0], np.zeros(result.shape[0]))) / correct_match
        print(f"Average Num Inliners: {ave_num_inliers}")
        recall_list.append(recall)
        inliers_list.append(ave_num_inliers)
    print(recall_list)
    average_recall = sum(recall_list) / len(recall_list)
    print(f"All 8 scene, average recall: {average_recall}%")
    average_inliers = sum(inliers_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers: {average_inliers}")
