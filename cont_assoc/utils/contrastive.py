import math
import MinkowskiEngine as ME
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
import torch

from cont_assoc.utils.kalman_filter import KalmanBoxTracker
import cont_assoc.utils.tracking as t

class PositionalEncoder(torch.nn.Module):
    # out_dim = in_dimnesionality * (2 * bands)
    def __init__(self, max_freq, feat_size, dimensionality, base=2):
        super().__init__()
        self.max_freq  = max_freq
        self.dimensionality = dimensionality
        self.num_bands = math.floor(feat_size/dimensionality/2)
        self.base = base
        pad = feat_size - self.num_bands*2 * dimensionality
        self.zero_pad = torch.nn.ZeroPad2d((pad,0,0,0))#left padding

    def forward(self, x):
        x = x/100
        x = x.unsqueeze(-1)
        device = x.device
        dtype = x.dtype

        scales = torch.logspace(0., math.log(
            self.max_freq / 2) / math.log(self.base), self.num_bands, base=self.base, device=device, dtype=dtype)
        # Fancy reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * math.pi
        x = torch.cat([x.sin(),x.cos()],dim=-1)
        x = x.flatten(1)
        enc = self.zero_pad(x)
        return enc

def group_instances(gt_coors, pt_raw_feat, ins_pred):
    coordinates = []
    features = []
    n_instances = []
    ins_ids = []
    for i in range(len(gt_coors)):#for every scan in the batch
        _coors = []
        _feats = []
        _ids = []
        pt_coors = gt_coors[i]#get point coordinates
        feat = pt_raw_feat[i].numpy()#get point features
        #get instance ids
        pt_ins_id = ins_pred[i]
        valid = pt_ins_id != 0 #ignore id=0
        ids, n_ids = np.unique(pt_ins_id[valid],return_counts=True)
        n_ins = 0
        for ii in range(len(ids)):#iterate over all instances
            if n_ids[ii] > 30:#filter too small instances
                pt_idx = np.where(pt_ins_id==ids[ii])[0]
                coors = torch.tensor(pt_coors[pt_idx],device='cuda')
                feats = torch.tensor(feat[pt_idx],device='cuda')
                _coors.extend([coors])
                _feats.extend([feats])
                _ids.extend([ids[ii]])
                n_ins += 1
        coordinates.append(_coors)
        features.append(_feats)
        n_instances.append(n_ins)
        ins_ids.append(_ids)
    return coordinates, features, n_instances, ins_ids, ins_pred

def update_ids(ids, id_assoc):
    for i in range(len(id_assoc)):#for each scan
        #new instance id_assoc[i][j][1] should be associated with previous instance id_assoc[0]
        if len(id_assoc[i]) > 0:
            for j in range(len(id_assoc[i])):
                #get index of that id in the input
                ind = np.argwhere(ids[i]==id_assoc[i][j][1])[0][0]
                # if ids[i][ind] != id_assoc[i][j][0]:
                    # print ("wasn't correct")
                ids[i][ind] = id_assoc[i][j][0]
    return ids

def fix_batches(ins_ids, features, coordinates, coordinates_T):
    new_feats = []
    new_coors = []
    new_coors_T = []
    for i in range(len(ins_ids)):
        c = 0
        if len(ins_ids[i]) == 0:
            new_feats.append([])
            new_coors.append([])
            new_coors_T.append([])
        else:
            new_feats.append(features[c])
            new_coors.append(coordinates[c])
            new_coors_T.append(coordinates_T[c])
            c += 1
    return new_feats, new_coors, new_coors_T
