import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule
import cont_assoc.models.blocks as blocks
import cont_assoc.models.panoptic_models as p_models
import cont_assoc.models.contrastive_models as c_models
import cont_assoc.utils.predict as pred
import cont_assoc.utils.contrastive as cont
import cont_assoc.utils.testing as testing
from cont_assoc.utils.evaluate_panoptic import PanopticKittiEvaluator
from cont_assoc.utils.evaluate_4dpanoptic import PanopticKitti4DEvaluator


class PS4D(LightningModule):
    def __init__(self, ps_cfg, tr_cfg):
        super().__init__()
        self.ps_cfg = ps_cfg
        self.tr_cfg = tr_cfg

        self.panoptic_model = p_models.PanopticCylinder(ps_cfg)
        self.tracking_head = c_models.ContrastiveTracking(tr_cfg)

        self.evaluator4D = PanopticKitti4DEvaluator(cfg=ps_cfg)

        self.last_ins_id = 0

    def load_state_dicts(self, ps_dict, tr_dict):
        self.tracking_head.load_state_dict(tr_dict)
        self.panoptic_model.load_state_dict(ps_dict)

    def merge_predictions(self, x, sem_logits, pred_offsets, pt_ins_feat):
        pt_sem_pred = pred.sem_voxel2point(sem_logits, x)
        clust_bandwidth = self.ps_cfg.MODEL.POST_PROCESSING.BANDWIDTH
        ins_pred = pred.cluster_ins(pt_sem_pred, pt_ins_feat, pred_offsets, x,
                                    clust_bandwidth, self.last_ins_id)
        sem_pred = pred.majority_voting(pt_sem_pred, ins_pred)

        return sem_pred, ins_pred

    def get_ins_feat(self, x, ins_pred, raw_features):
        #Group points into instances
        pt_raw_feat = pred.feat_voxel2point(raw_features,x)
        pt_coordinates = x['pt_cart_xyz']

        coordinates, features, n_instances, ins_ids, ins_pred = cont.group_instances(pt_coordinates, pt_raw_feat, ins_pred)

        #Discard scans without instances
        features = [x for x in features if len(x)!=0]
        coordinates = [x for x in coordinates if len(x)!=0]

        if len(features)==0:#don't run tracking head if no ins
            # return [], [], [], ins_pred
            return [], [], [], ins_pred, {}

        #Get per-instance feature
        tracking_input = {'pt_features':features,'pt_coors':coordinates}

        ins_feat = self.tracking_head(tracking_input)

        if len(coordinates) != len(ins_ids):
            #scans without instances
            new_feats, new_coors = cont.fix_batches(ins_ids, features, coordinates)
            tracking_input = {'pt_features':new_feats,'pt_coors':new_coors}

        return ins_feat, n_instances, ins_ids, ins_pred, tracking_input

    def track(self, ins_pred, ins_feat, n_instances, ins_ids, tr_input, poses):
        #Separate instances of different scans
        points = tr_input['pt_coors']
        features = tr_input['pt_features']
        ins_feat = torch.split(ins_feat, n_instances)
        poses = [[p] for p in poses]

        #Instance IDs association
        ins_pred = self.tracking_head.AssocModule.associate(ins_pred, ins_feat,
                                                            points, features,
                                                            poses, ins_ids)

        self.last_ins_id = self.tracking_head.AssocModule.get_last_id()
        self.tracking_head.AssocModule.update_last_id(self.last_ins_id)

        return ins_pred

    def forward(self, x):
        sem_logits, pred_offsets, pt_ins_feat, raw_features = self.panoptic_model(x)
        sem_pred, ins_pred = self.merge_predictions(x, sem_logits, pred_offsets, pt_ins_feat)
        ins_feat, n_ins, ins_ids, ins_pred, tracking_input = self.get_ins_feat(x, ins_pred, raw_features)

        #if no instances, don't track
        if len(ins_feat)!=0:
            ins_pred = self.track(ins_pred, ins_feat, n_ins, ins_ids, tracking_input, x['pose'])
        return sem_pred, ins_pred

    def test_step(self, batch, batch_idx):
        x = batch
        sem_pred, ins_pred = self(x)

        if 'RESULTS_DIR' in self.ps_cfg:
            results_dir = self.ps_cfg.RESULTS_DIR
            class_inv_lut = self.panoptic_model.evaluator.get_class_inv_lut()
            testing.save_results(sem_pred, ins_pred, results_dir, x, class_inv_lut)

        if 'UPDATE_METRICS' in self.ps_cfg:
            self.panoptic_model.evaluator.update(sem_pred, ins_pred, x)
            self.evaluator4D.update(sem_pred, ins_pred, x)
