import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
import pickle
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from tqdm import tqdm
from scipy import stats as s

class SemanticKittiModule(LightningDataModule):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.verbose = verbose

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        train_split = 'train'
        if 'SAVE_FEATURES' in self.cfg.keys():
            val_split = 'train'
            print("using train split as val split")
        else:
            val_split = 'valid'
        test_split = 'test'

        ########## Point dataset splits
        train_pt_dataset = SemanticKitti(
            self.cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            split = train_split
        )

        val_pt_dataset = SemanticKitti(
            self.cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            split = val_split
        )

        test_pt_dataset = SemanticKitti(
            self.cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            split = test_split
        )

        ########## Voxel spherical dataset splits
        self.train_set = CylindricalSemanticKitti(
            train_pt_dataset,
            grid_size = self.cfg.DATA_CONFIG.DATALOADER.GRID_SIZE,
            ignore_label = self.cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL,
            fixed_volume_space = self.cfg.DATA_CONFIG.DATALOADER.FIXED_VOLUME_SPACE,
        )

        self.val_set = CylindricalSemanticKitti(
            val_pt_dataset,
            grid_size = self.cfg.DATA_CONFIG.DATALOADER.GRID_SIZE,
            ignore_label = self.cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL,
            fixed_volume_space = self.cfg.DATA_CONFIG.DATALOADER.FIXED_VOLUME_SPACE,
        )

        self.test_set = CylindricalSemanticKitti(
            test_pt_dataset,
            grid_size = self.cfg.DATA_CONFIG.DATALOADER.GRID_SIZE,
            ignore_label = self.cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL,
            fixed_volume_space = self.cfg.DATA_CONFIG.DATALOADER.FIXED_VOLUME_SPACE,
        )

    def train_dataloader(self):
        self.train_loader = DataLoader(
            dataset = self.train_set,
            batch_size = self.cfg.EVAL.BATCH_SIZE,
            collate_fn = collate_fn_BEV,
            shuffle = self.cfg.DATA_CONFIG.DATALOADER.SHUFFLE,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = True,
            drop_last = False,
            timeout = 0
        )
        self.train_iter = iter(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        self.valid_loader = DataLoader(
            dataset = self.val_set,
            batch_size = self.cfg.EVAL.BATCH_SIZE,
            collate_fn = collate_fn_BEV,
            shuffle = False,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = True,
            drop_last = False,
            timeout = 0
        )
        self.valid_iter = iter(self.valid_loader)
        return self.valid_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
            dataset = self.test_set,
            batch_size = self.cfg.EVAL.BATCH_SIZE,
            collate_fn = collate_fn_BEV,
            shuffle = False,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = True,
            drop_last = False,
            timeout = 0
        )
        self.test_iter = iter(self.test_loader)
        return self.test_loader

class SemanticKitti(Dataset):
    def __init__(self, data_path, split='train', seq=None):
        with open("datasets/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        SemKITTI_label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
        self.learning_map = semkittiyaml['learning_map']
        self.split = split
        split = semkittiyaml['split'][self.split]

        self.im_idx = []
        pose_files = []
        calib_files = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'velodyne']))
            pose_files.append(absoluteDirPath(data_path+str(i_folder).zfill(2)+'/poses.txt'))
            calib_files.append(absoluteDirPath(data_path+str(i_folder).zfill(2)+'/calib.txt'))

        self.im_idx.sort()
        self.poses = load_poses(pose_files, calib_files)

        self.things = ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'person',
                       'bicyclist', 'motorcyclist']
        self.stuff = ['road', 'sidewalk', 'parking', 'other-ground', 'building',
                      'vegetation', 'trunk', 'terrain', 'fence', 'pole', 'traffic-sign']
        self.things_ids = []
        for i in sorted(list(semkittiyaml['labels'].keys())):
            if SemKITTI_label_name[semkittiyaml['learning_map'][i]] in self.things:
                self.things_ids.append(i)

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
            sem_labels = annotated_data
            ins_labels = annotated_data
            valid = annotated_data
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
            sem_labels = annotated_data & 0xFFFF #delete high 16 digits binary
            ins_labels = annotated_data
            valid = np.isin(sem_labels, self.things_ids).reshape(-1) # use 0 to filter out valid indexes is enough
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)
        data_tuple = (raw_data[:,:3], sem_labels.astype(np.uint8))
        data_tuple += (raw_data[:,3],)#ref
        data_tuple += (ins_labels, valid)#ins ids
        data_tuple += (self.im_idx[index],)#filename
        data_tuple += (self.poses[index],)#pose
        return data_tuple

class CylindricalSemanticKitti(Dataset):
  def __init__(self, in_dataset, grid_size, min_rad=-np.pi/4, max_rad=np.pi/4,
               ignore_label = 255, fixed_volume_space= False,
               max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3],
               center_type='Axis_center'):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        self.center_type = center_type

  def __len__(self):
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        data = self.point_cloud_dataset[index]
        if len(data) == 6:
            xyz,labels,sig,ins_labels,valid,pcd_fname = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 7:
            xyz,labels,sig,ins_labels,valid,pcd_fname,pose = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1) # (size-1) could directly get index starting from 0, very convenient

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int) # point-wise grid index

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        # process valid voxel labels
        valid_voxels, vox_to_point = np.unique(grid_ind, return_inverse=True, axis=0)
        voxel_labels = np.ones(valid_voxels.shape[0],dtype = np.uint8)*self.ignore_label

        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,2],grid_ind[:,1],grid_ind[:,0])),:] #same order as coordinates to create sparse tensor when using np.unique

        voxel_labels = nb_get_voxel_labels(np.copy(voxel_labels),label_voxel_pair) #get valid voxel labels
        voxel_labels = voxel_labels.reshape(1,voxel_labels.shape[0]) #add batch dimension

        data_tuple = (voxel_position,voxel_labels)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        #point features
        return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)

        # (grid-wise coor, grid-wise sem label, point-wise grid index, indices voxel to point, point-wise sem label, [relative polar coor(3), polar coor(3), cat coor(2), ref signal(1)])
        data_tuple += (grid_ind, vox_to_point,labels,return_fea)
        offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
        offsets = nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, self.center_type)

        if len(data) == 6:
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname) # plus (point-wise instance label, point-wise center offset)

        if len(data) == 7:
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname, pose) # plus (point-wise instance label, point-wise center offset)

        return data_tuple

def calc_xyz_middle(xyz):
    return np.array([
        (np.max(xyz[:, 0]) + np.min(xyz[:, 0])) / 2.0,
        (np.max(xyz[:, 1]) + np.min(xyz[:, 1])) / 2.0,
        (np.max(xyz[:, 2]) + np.min(xyz[:, 2])) / 2.0
    ], dtype=np.float32)

things_ids = set([10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259])

# @nb.jit
def nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, center_type):
    # ins_num = np.max(ins_labels) + 1
    # for i in range(1, ins_num):
    for i in np.unique(ins_labels):
        if (i & 0xFFFF) not in things_ids:
            continue
        i_indices = (ins_labels == i).reshape(-1)
        xyz_i = xyz[i_indices]
        if xyz_i.shape[0] <= 0:
            continue
        if center_type == 'Axis_center':
            mean_xyz = calc_xyz_middle(xyz_i)
        elif center_type == 'Mass_center':
            mean_xyz = np.mean(xyz_i, axis=0)
        offsets[i_indices] = mean_xyz - xyz_i
    return offsets

@nb.jit('u1[:](u1[:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_get_voxel_labels(voxel_labels,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    voxel_ind = 0
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            voxel_labels[voxel_ind] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
            voxel_ind += 1
        counter[sorted_label_voxel_pair[i,3]] += 1
    voxel_labels[voxel_ind] = np.argmax(counter)
    return voxel_labels

def collate_fn_BEV(data): # stack along batch dimension
    grid_ind_all = [d[2] for d in data]
    label_all = [d[1] for d in data]
    pt_label_all = [d[3] for d in data]
    data2stack=np.stack([d[0] for d in data]).astype(np.float32) # grid-wise coor
    label2stack = [d[1] for d in data]                           # grid-wise sem label
    grid_ind_stack = [d[2] for d in data]                        # point-wise grid index
    vox_to_point = [d[3] for d in data]                          # indices mapping voxel-to-point
    point_label = [d[4] for d in data]                           # point-wise sem label
    xyz = [d[5] for d in data]                                   # point-wise coor

    pt_ins_labels = [d[6] for d in data]                         # point-wise instance label
    pt_offsets = [d[7] for d in data]                            # point-wise center offset
    pt_valid = [d[8] for d in data]                              # point-wise indicator for foreground points
    pt_cart_xyz = [d[9] for d in data]                           # point-wise cart coor
    filename = [d[10] for d in data]                             # scan filename
    pose = [d[11] for d in data]                                 # pose of the scan

    return {
        'vox_coor': torch.from_numpy(data2stack),
        'vox_labels': label2stack,
        'grid': grid_ind_stack,
        'vox2point_idx' : vox_to_point,
        'pt_labs': point_label,
        'pt_fea': xyz,
        'pt_ins_labels': pt_ins_labels,
        'pt_offsets': pt_offsets,
        'pt_valid': pt_valid,
        'pt_cart_xyz': pt_cart_xyz,
        'pcd_fname': filename,
        'pose': pose
    }

# Transformations between Cartesian and Polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def absoluteDirPath(directory):
    return os.path.abspath(directory)

def parse_calibration(filename):
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

def parse_poses(filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses

def load_poses(pose_files, calib_files):
    poses = []
    for i in range(len(pose_files)):
        calib = parse_calibration(calib_files[i])
        seq_poses_f64 = parse_poses(pose_files[i], calib)
        seq_poses = ([pose.astype(np.float32) for pose in seq_poses_f64])
        poses += seq_poses
    return poses
