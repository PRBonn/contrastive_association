import numpy as np
import os
import torch

def create_dirs(file_dir, test_set):
    split = 'val'
    if test_set:
        split = 'test'
    results_dir = join(file_dir, 'output', split, 'panoptic', 'sequences')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if test_set:
        for i in range(11,22):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
    else:
        sub_dir = os.path.join(results_dir, str(8).zfill(2), 'predictions')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
    return results_dir

def save_results(sem_preds, ins_preds, output_dir, batch, class_inv_lut):
    for i in range(len(sem_preds)):
        sem = sem_preds[i]
        ins = ins_preds[i]
        sem_inv = class_inv_lut[sem].astype(np.uint32)
        label = sem_inv.reshape(-1, 1) + ((ins.astype(np.uint32) << 16) & 0xFFFF0000).reshape(-1, 1)

        pcd_path = batch['pcd_fname'][i]
        seq = pcd_path.split('/')[-3]
        pcd_fname = pcd_path.split('/')[-1].split('.')[-2]+'.label'
        fname = os.path.join(output_dir, seq, 'predictions', pcd_fname)
        label.reshape(-1).astype(np.uint32).tofile(fname)
