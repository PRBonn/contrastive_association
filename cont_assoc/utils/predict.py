import numpy as np
import os
from scipy import stats as s
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.cluster import MeanShift
import torch

from cont_assoc.utils.kalman_filter import KalmanBoxTracker
import cont_assoc.utils.tracking as t
import cont_assoc.utils.contrastive as cont

######Clustering

def cluster_ins(sem_preds, pt_ins_feat, pred_offsets, inputs, bandwidth, last_ins_id):
    valid_xentropy_ids = [1, 4, 2, 3, 5, 6, 7, 8]
    grid_ind = inputs['grid']
    pt_cart_xyz = inputs['pt_cart_xyz']
    pt_pred_offsets = [pred_offsets[i].detach().cpu().numpy().reshape(-1, 3) for i in range(len(pred_offsets))] #x,y,z
    pt_ins_feat = [pt_ins_feat[i].detach().cpu().numpy() for i in range(len(pt_ins_feat))]
    pt_pred_valid = []
    for i in range(len(grid_ind)):
        pt_pred_valid.append(np.isin(sem_preds[i], valid_xentropy_ids).reshape(-1))
    pred_ins_ids_list = []
    for i in range(len(pt_cart_xyz)):
        i_clustered_ins_ids  = meanshift_cluster(pt_cart_xyz[i] + pt_pred_offsets[i], pt_pred_valid[i], bandwidth)
        thing_ind = np.where(i_clustered_ins_ids != 0)
        i_clustered_ins_ids[thing_ind] += last_ins_id + 1
        last_ins_id = max(i_clustered_ins_ids)
        pred_ins_ids_list.append(i_clustered_ins_ids)
    return pred_ins_ids_list


def meanshift_cluster(shifted_pcd, valid, bandwidth=1.0):
    shift_dim = shifted_pcd.shape[1]
    clustered_ins_ids = np.zeros(shifted_pcd.shape[0], dtype=np.int32)
    valid_shifts = shifted_pcd[valid, :].reshape(-1, shift_dim) if valid is not None else shifted_pcd
    if valid_shifts.shape[0] == 0:
        return clustered_ins_ids
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    try:
        ms.fit(valid_shifts)
    except Exception as e:
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(valid_shifts)
        print("\nException: {}.".format(e))
        print("Disable bin_seeding.")
    labels = ms.labels_ + 1
    assert np.min(labels) > 0
    if valid is not None:
        clustered_ins_ids[valid] = labels
        return clustered_ins_ids
    else:
        return labels

def sem_voxel2point(sem_logits, inputs):
    vox_pred = torch.argmax(sem_logits.features, dim=1)
    vox_pred = vox_pred.cpu().detach().numpy()
    vox2point_idx = inputs['vox2point_idx'] #indices mapping voxel to point
    n_valid = [inputs['vox_labels'][i].shape[1] for i in range(len(inputs['vox_labels']))] #n of valid voxels
    vox_range = np.insert(np.add.accumulate(n_valid),0,0) #indices to acces per-batch valid voxels
    point_pred = [vox_pred[vox_range[i]:vox_range[i+1]][vox2point_idx[i]] for i in range(len(vox2point_idx))]
    return point_pred

def feat_voxel2point(features, inputs):
    vox_feat = features.features.cpu()
    vox2point_idx = inputs['vox2point_idx'] #indices mapping voxel to point
    n_valid = [inputs['vox_labels'][i].shape[1] for i in range(len(inputs['vox_labels']))] #n of valid voxels
    vox_range = np.insert(np.add.accumulate(n_valid),0,0) #indices to acces per-batch valid voxels
    point_feat = [vox_feat[vox_range[i]:vox_range[i+1]][vox2point_idx[i]] for i in range(len(vox2point_idx))]
    return point_feat

def majority_voting(sem_preds, pred_ins_ids):
    merged_sem_preds = []
    for i in range(len(sem_preds)): #all scans in the batch
        sem = sem_preds[i].copy()
        ins_ids = np.unique(pred_ins_ids[i])
        for _id in ins_ids: #all instances
            if _id == 0: # ignore stuff
                continue
            ind = np.where(pred_ins_ids[i] == _id) #indices of instances with _id
            # if ind[0].shape[0] < 30: #filter instances with few points
                # continue
            #majority voting
            (classes, cnts) = np.unique(sem[ind], return_counts=True)
            inst_class = classes[np.argmax(cnts)]
            sem[ind] = inst_class
        merged_sem_preds.append(sem)
    return merged_sem_preds

