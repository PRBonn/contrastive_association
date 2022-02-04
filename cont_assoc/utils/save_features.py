import numpy as np
import os
import torch
from shutil import copyfile

#Save point features for contrastive approach
def save_features(x, raw_feat, sem_pred, ins_pred, save_preds):
    for i in range(len(x['pcd_fname'])):
        _ids = []
        _sem_labels = []
        _n_pts = []
        _coors = []
        _feats = []
        fname = x['pcd_fname'][i][-10:-4]
        seq = x['pcd_fname'][i][-22:-20]
        pt_coors = x['pt_cart_xyz'][i]
        feat = raw_feat[i].numpy()
        if save_preds:
            sem = sem_pred[i]
            ins =  ins_pred[i]
            valid = ins != 0
            seq_path = 'data/validation_predictions/sequences/'+seq+'/'
            max_pt = 30
        else:
            valid = x['pt_valid'][i]
            sem = x['pt_labs'][i]
            ins =  x['pt_ins_labels'][i]
            seq_path = 'data/instance_features/sequences/'+seq+'/'
            max_pt = 10
        ids, n_ids = np.unique(ins[valid],return_counts=True)
        for ii in range(len(ids)):
            if n_ids[ii] <= max_pt:
                continue
            pt_idx = np.where(ins==ids[ii])[0]
            coors = torch.tensor(pt_coors[pt_idx])
            sem_label = np.unique(sem[pt_idx])
            features = torch.tensor(feat[pt_idx])
            n_pt = n_ids[ii]
            _ids.extend([ids[ii]])
            _sem_labels.extend(sem_label)
            _n_pts.extend([n_pt])
            _coors.extend([coors])
            _feats.extend([features])
        filename = seq_path + 'scans/' + fname
        if not os.path.exists(seq_path):
            os.makedirs(seq_path+'scans/')
            orig_seq_path = x['pcd_fname'][i][:-19]
            #copy txt files
            copyfile(orig_seq_path + 'poses.txt', seq_path + 'poses.txt')
            copyfile(orig_seq_path + 'calib.txt', seq_path + 'calib.txt')
            #create empty file
            f = open(seq_path + 'empty.txt','w')
            f.close()
        if not save_preds:
            #dont save if no instances: len(ids) == 0
            if len(_ids) == 0:
                f = open(seq_path + 'empty.txt','a')
                f.write(fname+'\n')
                f.close()
                continue
        if save_preds:
            np_instances = np.array([seq,fname,_ids,_sem_labels,_n_pts,_coors,_feats,sem_pred[i],ins_pred[i],x['pcd_fname'][i],x['pt_labs'][i],x['pt_ins_labels'][i]],dtype=object)
        else:
            np_instances = np.array([seq,fname,_ids,_sem_labels,_n_pts,_coors,_feats],dtype=object)

        np.save(filename,np_instances,allow_pickle=True)
