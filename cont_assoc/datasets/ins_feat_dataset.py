import os
import numpy as np
import torch
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
import cont_assoc.utils.pcd_augmentations as aug
import cont_assoc.utils.tracking as tr

class InstanceFeaturesModule(LightningDataModule):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.verbose = verbose

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if 'ONLY_SEQ' in self.cfg.TRAIN.keys():
            only_seq = self.cfg.TRAIN.ONLY_SEQ
            val_dataset_path = self.cfg.DATA_CONFIG.DATASET_PRED_PATH + '/sequences/'
        else:
            only_seq = None
            val_dataset_path = self.cfg.DATA_CONFIG.VAL_PRED_PATH + '/sequences/'

        pos_scans = self.cfg.TRAIN.POS_SCANS

        train_dataset = InstanceFeatures(
            self.cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            split = 'train',
            pos_scans = pos_scans,
            seq = only_seq,
            augmentations = self.cfg.DATA_CONFIG.DATALOADER.AUGMENTATION,
            r_pos_scans = self.cfg.TRAIN.RANDOM_POS_SCANS
        )

        val_dataset = InstanceFeatures(
            val_dataset_path,
            split = 'valid',
            pos_scans = 0,#only get the scan
            seq = only_seq
        )

        test_dataset = InstanceFeatures(
            # self.cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            val_dataset_path,
            # split = 'test',
            # DON'T HAVE A TEST SET NOW
            split = 'valid',
            pos_scans = 0,
            seq = only_seq
        )

        collate_function = collateInstances()

        ########## Generate dataloaders and iterables
        self.train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = self.cfg.TRAIN.BATCH_SIZE,
            collate_fn = collate_function,
            shuffle = self.cfg.DATA_CONFIG.DATALOADER.SHUFFLE,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = False,
            drop_last = False,
            timeout = 0
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset = val_dataset,
            batch_size = self.cfg.TRAIN.BATCH_SIZE,
            collate_fn = collate_function,
            shuffle = False,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = False,
            drop_last = False,
            timeout = 0
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = self.cfg.TRAIN.BATCH_SIZE,
            collate_fn = collate_function,
            shuffle = False,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = False,
            drop_last = False,
            timeout = 0
        )
        self.test_iter = iter(self.test_loader)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


class InstanceFeatures(Dataset):
    def __init__(self, data_path, pos_scans, split='train', seq=None, augmentations=None, r_pos_scans=False):
        self.pos_scans = pos_scans
        self.data_path = data_path
        self.aug = augmentations
        self.random_pos_scans = r_pos_scans
        with open("datasets/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        SemKITTI_label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
        self.learning_map = semkittiyaml['learning_map']
        self.split = split
        split = semkittiyaml['split'][self.split]
        if seq is not None:
            split = [seq]

        self.im_idx = []
        pose_files = []
        calib_files = []
        empty_files = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'scans']))
            pose_files.append(absoluteDirPath(data_path+str(i_folder).zfill(2)+'/poses.txt'))
            calib_files.append(absoluteDirPath(data_path+str(i_folder).zfill(2)+'/calib.txt'))
            empty_files.append(absoluteDirPath(data_path+str(i_folder).zfill(2)+'/empty.txt'))

        self.im_idx.sort()
        self.poses = load_poses(pose_files, calib_files, empty_files)

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        ids = []
        sem_labels = np.array([])
        pos_labels = np.array([],dtype=np.int)
        n_pts = []
        pos_enc = []
        pt_coors = []
        pt_coors_T = []
        pt_features = []
        poses = []
        pt_sem_preds = []#for validation predictions
        pt_ins_preds = []#for validation predictions

        # select random scan based on index
        fname = self.im_idx[index]
        scan = int(fname[-10:-4])
        seq = fname[-19:-17]
        scans = [scan]
        pos_idx = [index]
        #select all scans in the sequence
        if self.random_pos_scans:
            #random number [1,pos_scans]
            n_scans = np.random.randint(0,self.pos_scans+1)+1
            #if using previous and future scans
            pair = bool(np.random.randint(0,2))
        else:
            n_scans = self.pos_scans + 1
            pair = True
        for i in range(1,n_scans):
            if pair:
                if scan-i >= 0:
                    scans.append(scan-i)
                    pos_idx.append(index-i)
            scans.append(scan+i)
            pos_idx.append(index+i)
        scans.sort()
        pos_idx.sort()

        prev_scan = 0
        if scans[0] > 0:
            prev_scan = scans[0] - 1

        seq_first_pose = []

        for i in range(len(scans)):
            if i == 0:
                prev_scan_path = absoluteDirPath(self.data_path+seq+'/scans/'+str(prev_scan).zfill(6)+'.npy')
                if os.path.exists(prev_scan_path):
                    prev_data = np.load(prev_scan_path,allow_pickle=True)
                    prev_pose = self.poses[prev_scan]
                    prev_ids = prev_data[2]
                    prev_coors = prev_data[5]
                    prev_coors_T = apply_pose(prev_coors,prev_pose)
                else:
                    prev_pose = []
                    prev_ids = []
                    prev_coors = []
                    prev_coors_T = []
            scan_path = absoluteDirPath(self.data_path+seq+'/scans/'+str(scans[i]).zfill(6)+'.npy')
            if os.path.exists(scan_path):
                #Check max number of points to avoid running out of memory
                if sum(n_pts) > 100000: #max n_pts=5e5, bs=4 --> max n_pts=125k
                    break
                pose = self.poses[pos_idx[i]]
                if len(seq_first_pose) == 0:
                    seq_first_pose = pose
                data = np.load(scan_path,allow_pickle=True)
                _ids = data[2]
                _pos_lab = _ids
                _sem_lab = data[3]
                _n_pt = data[4]
                _coors = data[5]
                _feats = data[6]

                if self.aug is not None and self.aug.DO_AUG == True:
                    _coors, _feats, _n_pt = self.apply_augmentations(_coors, _feats, _n_pt)

                _coors_T = apply_pose(_coors,pose)

                #shift coors to local frame of first scan in the sequence
                if self.split == 'train':
                    _coors = apply_pose(_coors_T, np.linalg.inv(seq_first_pose))

                ids.extend(_ids)
                sem_labels = np.append(sem_labels,_sem_lab)
                pos_labels = np.append(pos_labels,_pos_lab)
                n_pts.extend(_n_pt)
                pt_coors.extend(_coors)
                pt_coors_T.extend(_coors_T)
                pt_features.extend(_feats)
                poses.append(pose)

                if data.shape[0] == 12: #validation predictions
                    pt_sem_pred = data[7]
                    pt_ins_pred = data[8]
                    pcd_fname = data[9]
                    pt_labs = data[10]
                    pt_ins_labels = data[11]

                prev_coors = _coors
                prev_coors_T = _coors_T
                prev_ids = _ids

        data_tuple = (ids, sem_labels, pos_labels, n_pts, pt_coors, pt_coors_T, pt_features, poses)
        if data.shape[0] == 12:
            data_tuple += (pt_sem_pred,)
            data_tuple += (pt_ins_pred,)
            data_tuple += (pcd_fname,)
            data_tuple += (pt_labs,)
            data_tuple += (pt_ins_labels,)

        return data_tuple

    def apply_augmentations(self, coors, feats, n_pts):
        augs = self.aug
        tr_coors = []
        tr_feats = []
        tr_n_pt = []
        for i in range(len(coors)):
            #apply transformations to each instance
            tc = coors[i].numpy()
            tf = feats[i].numpy()
            if augs.JITTER:
                tc = aug.jitter_point_cloud(tc)
            if augs.POINTS:
                tc, tf = aug.random_point_dropout(tc,tf)
            if augs.CUBOIDS:
                tc, tf = aug.random_drop_n_cuboids(tc,tf)
            if augs.PLANE:
                tc, tf = aug.random_plane_dropout(tc,tf)
            if augs.CONTOUR:
                tc, tf = aug.contour_dropout(tc,tf)
            if tc.shape[0] >= 10:
                tr_coors.append(torch.tensor(tc,dtype=torch.float))
                tr_feats.append(torch.tensor(tf,dtype=torch.float))
                tr_n_pt.append(len(tc))
            else:
                tr_coors.append(coors[i])
                tr_feats.append(feats[i])
                tr_n_pt.append(n_pts[i])
        return tr_coors, tr_feats, tr_n_pt

class collateInstances:

    def __init__(self):
        pass

    def __call__(self, data):
        ids = [d[0] for d in data]
        sem_labels = [d[1] for d in data]
        pos_labels = [d[2] for d in data]
        n_pts = [d[3] for d in data]
        pt_coors = [d[4] for d in data]
        pt_coors_T = [d[5] for d in data]
        pt_features = [d[6] for d in data]
        pose = [d[7] for d in data]

        out_dict =  {                       #for each instance:
            'id': ids,                      #instance id
            'sem_label': sem_labels,        #semantic label
            'pos_label': pos_labels,        #positive label: to consider as positive example
            'n_pts' : n_pts,                #number of points depicting the instance
            'pt_coors' : pt_coors,          #xyz coordinates for each point [n_pts,3]
            'pt_coors_T' : pt_coors_T,      #global points coordinates [n_pts,3]
            'pt_features' : pt_features,    #features for every point [n_pts,128]
            'pose' : pose,                  #scan center global position
            }

        if len(data[0]) == 13: #validation predictions
            pt_sem_pred = [d[8] for d in data]
            pt_ins_pred = [d[9] for d in data]
            pcd_fname = [d[10] for d in data]
            pt_labs = [d[11] for d in data]
            pt_ins_labels = [d[12] for d in data]
            out_dict['pt_sem_pred'] = pt_sem_pred        #sem preds for all points in the scan
            out_dict['pt_ins_pred'] = pt_ins_pred        #ins preds for all points in the scan
            out_dict['pcd_fname'] = pcd_fname            #filename
            out_dict['pt_labs'] = pt_labs                #per point sem label
            out_dict['pt_ins_labels'] = pt_ins_labels    #per point instance label
        return out_dict

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

def get_empty(filename):
    empty_list = []
    empty_file = open(filename)
    for line in empty_file:
        empty_list.append(int(line.strip()))
    empty_file.close()
    return empty_list

def load_poses(pose_files, calib_files, empty_files):
    poses = []
    #go through every file and get all poses
    #add them to match im_idx
    for i in range(len(pose_files)):
        empty = get_empty(empty_files[i])
        calib = parse_calibration(calib_files[i])
        seq_poses_f64 = parse_poses(pose_files[i], calib)
        seq_poses = ([seq_poses_f64[i].astype(np.float32) for i in range(len(seq_poses_f64)) if i not in empty])
        poses += seq_poses
    return poses

def apply_pose(points, pose):
    shifted_points = []
    for i in range(len(points)):#for each instance
        hpoints = np.hstack((points[i].numpy()[:, :3], np.ones_like(points[i].numpy()[:, :1])))
        shifted_points.append(torch.tensor(np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:,:3]))
    return shifted_points
