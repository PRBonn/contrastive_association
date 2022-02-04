import math
import MinkowskiEngine as ME
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
import torch

from cont_assoc.utils.kalman_filter import KalmanBoxTracker
import cont_assoc.utils.tracking as t
import cont_assoc.utils.contrastive as cont

class AssociationModule():
    def __init__(self, weights, thresholds, enc, pos_enc, use_poses):
        super().__init__()
        self.assoc_w = weights
        self.assoc_T = thresholds
        self.use_poses = use_poses
        self.pos_encoder = pos_enc
        self.encoder = enc
        self.tr_ins = {}
        self.last_ins_id = 0
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def clear(self):
        self.tr_ins = {}
        self.last_ins_id = 0

    def update_last_id(self, l_id):
        self.last_ins_id = l_id

    def get_last_id(self):
        return self.last_ins_id


    def associate(self, ins_preds, ins_feat, pt_coors, pt_feat, poses, ins_ids):
        new_ins_preds = []
        for i in range(len(ins_feat)): #for every scan in the batch
            if len(ins_feat[i]) == 0: #no instances
                new_ins_preds.append(ins_preds[i])
                continue
            ins_pred = ins_preds[i]
            curr_ins = self.init_curr_ins(ins_pred, ins_feat[i], pt_coors[i],
                                          poses[i][0],pt_feat[i],ins_ids[i])

            if len(self.tr_ins) > 0:
                inv_pose = np.linalg.inv(poses[i][0])
                self.predict_poses(inv_pose)
                cost_matrix, assoc_pairs = self.get_associations(self.tr_ins, curr_ins)
                curr_ins = self.perform_associations(curr_ins, assoc_pairs, ins_pred)
            self.add_non_matching_ins(curr_ins, ins_pred)
            self.kill_old_ins()

            if len(self.tr_ins) != 0: self.last_ins_id = max(self.tr_ins)

            ins_pred = self.clean_pred(ins_pred)
            new_ins_preds.append(ins_pred)
        return new_ins_preds

    def init_curr_ins(self, ins_pred, ins_feat, pt_coors, pose, pt_feats, ins_ids):
        curr_ins = {}
        for j in range(len(ins_ids)): #go over all instances in current scan
            ind = np.where(ins_pred == ins_ids[j])
            #filter instances with few points
            if ind[0].shape[0] < 30:
                ins_pred[ind] = 0
                continue
            #initialize object tracks in current frame
            _coors = pt_coors[j]
            if self.use_poses:
                _coors = self.apply_pose(pt_coors[j],pose)
            bbox, k_bbox = t.get_bbox_from_points(_coors.cpu().numpy())
            tracker = KalmanBoxTracker(k_bbox, ins_ids[j])
            curr_ins[ins_ids[j]] = {'life': 8,
                                   'feature': ins_feat[j],
                                   'pt_coors': _coors,
                                   'pt_feats': pt_feats[j],
                                   'kalman_bbox': k_bbox,
                                   'tracker': tracker}
        return curr_ins

    def predict_poses(self, inv_pose):
        for k in self.tr_ins.keys():
            points = self.tr_ins[k]['pt_coors']
            self.tr_ins[k]['kalman_bbox'] = self.tr_ins[k]['tracker'].predict()
            #Update positional encoding and features with poses
            if self.use_poses:
                #Transform from global to local coordinates
                t_points = self.apply_pose(points, inv_pose)
                #Create sparse tensor using new points coordinates
                pt_feat = self.tr_ins[k]['pt_feats']
                sparse = self.sparse_tensor(t_points, pt_feat, self.pos_encoder)
                #Update feature using contrastive network
                new_feat = self.encoder(sparse)
                self.tr_ins[k]['feature'] = new_feat.squeeze(0)

    def get_associations(self, prev_ins, curr_ins):
        dist_w, feat_w = self.assoc_w
        dist_T, feat_T = self.assoc_T
        cost_matrix = np.zeros((len(prev_ins), len(curr_ins)))
        prev_ids = []
        curr_ids = []
        for i, (id1, v1) in enumerate(prev_ins.items()):
            prev_ids.append(id1)
            for j, (id2, v2) in enumerate(curr_ins.items()):
                if i == 0: curr_ids.append(id2)

                cost_feature = 1 - self.cos(v2['feature'], v1['feature']).cpu().numpy()
                if cost_feature > feat_T: cost_feature = 1e8 - 1

                cost_dist = t.euclidean_dist(v2['kalman_bbox'], v1['kalman_bbox'])
                if cost_dist > dist_T: cost_dist = 1e8 - 1

                cost_matrix[i,j] = dist_w * cost_dist + feat_w * cost_feature

        idx1, idx2 = lsa(cost_matrix)
        assoc_pairs = []
        for i1, i2 in zip(idx1, idx2):
            if cost_matrix[i1][i2] < 1e8:
                assoc_pairs.append((prev_ids[i1], curr_ids[i2]))
        return cost_matrix, assoc_pairs

    def perform_associations(self, curr_ins, assoc_pairs, ins_pred):
        for prev_id, new_id in assoc_pairs:
            ins_ind = np.where((ins_pred == new_id))
            #assign consistent instance id to previous prediction
            ins_pred[ins_ind[0]] = prev_id

            #update tracked_instances with the ones in current frame
            self.tr_ins[prev_id]['life'] += 1
            self.tr_ins[prev_id]['feature'] = curr_ins[new_id]['feature']
            self.tr_ins[prev_id]['pt_coors'] = curr_ins[new_id]['pt_coors']
            self.tr_ins[prev_id]['pt_feats'] = curr_ins[new_id]['pt_feats']
            self.tr_ins[prev_id]['kalman_bbox'] = self.tr_ins[prev_id]['tracker'].get_state()
            self.tr_ins[prev_id]['tracker'].update(curr_ins[new_id]['kalman_bbox'], prev_id)

            del curr_ins[new_id] #remove already assigned instances
        return curr_ins

    def add_non_matching_ins(self, new_ins, ins_pred):
        id_cont = 0
        for _id, instance in new_ins.items():
            idx = np.where(ins_pred == _id)
            if idx[0].shape[0] < 30:
                continue
            #check if the id isn't already used
            if not _id in self.tr_ins:
                self.tr_ins[_id] = instance
            else:
                _id = self.last_ins_id + id_cont
                self.tr_ins[_id] = instance
                id_cont += 1

    def kill_old_ins(self):
        dont_track_ids = []
        for _id in self.tr_ins.keys():
            if self.tr_ins[_id]['life'] == 0:
                dont_track_ids.append(_id)
            else:
                self.tr_ins[_id]['life'] -= 1
        for _id in dont_track_ids:
            del self.tr_ins[_id]

    def clean_pred(self, ins_pred):
        for _id in np.unique(ins_pred):
            if _id == 0:
                continue
            valid_ind = np.argwhere(ins_pred == _id)[:, 0]
            if valid_ind.shape[0] < 30:
                ins_pred[valid_ind] = 0
        return ins_pred

    def sparse_tensor(self, pt_coors, pt_features, pos_encoder):
        pos_encoding = pos_encoder(pt_coors)
        pt_features = pt_features + pos_encoding
        c_, f_ = ME.utils.sparse_collate([pt_coors], [pt_features], dtype=torch.float32)
        sparse = ME.SparseTensor(features=f_.float(), coordinates=c_.int(),device='cuda')
        return sparse

    def apply_pose(self, points, pose):
        hpts = torch.hstack((points[:, :3], torch.ones_like(points[:, :1]))).type(torch.float64)
        t_pose = torch.tensor(pose,dtype=torch.float64,device='cuda').T
        tr_pts = torch.mm(hpts,t_pose)
        shifted_pts = tr_pts[:,:3]
        return shifted_pts
