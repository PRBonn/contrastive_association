import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule
import cont_assoc.models.blocks as blocks
import cont_assoc.utils.predict as pred
import cont_assoc.utils.testing as testing
import cont_assoc.utils.save_features as sf
from cont_assoc.utils.evaluate_panoptic import PanopticKittiEvaluator
from cont_assoc.utils.evaluate_4dpanoptic import PanopticKitti4DEvaluator


class PanopticCylinder(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.voxel_feature_extractor = blocks.VoxelFeatureExtractor(cfg)
        self.encoder = CylinderEncoder(cfg)
        self.decoder = CylinderDecoder(cfg)
        self.semantic_head = CylinderSemanticHead(cfg)
        self.instance_head = CylinderInstanceHead(cfg)

        self.evaluator = PanopticKittiEvaluator(cfg=cfg)

    def get_pq(self):
        return self.evaluator.get_mean_pq()

    def merge_predictions(self, x, sem_logits, pred_offsets, pt_ins_feat):
        #Sem labels
        pt_sem_pred = pred.sem_voxel2point(sem_logits, x)

        #Ins labels
        last_ins_id = 0
        clust_bandwidth = self.cfg.MODEL.POST_PROCESSING.BANDWIDTH
        ins_embedding_dim = self.cfg.MODEL.INS_HEAD.EMBEDDING_CHANNEL
        ins_pred = pred.cluster_ins(pt_sem_pred, pt_ins_feat, pred_offsets, x,
                                    clust_bandwidth, last_ins_id)
        #Majority voting
        sem_pred = pred.majority_voting(pt_sem_pred, ins_pred)

        return sem_pred, ins_pred

    def forward(self, x):
        coordinates, voxel_features = self.voxel_feature_extractor(x)
        encoding, skips = self.encoder(voxel_features, coordinates, len(x['grid']))
        semantic_feat, instance_feat = self.decoder(encoding, skips)
        semantic_logits = self.semantic_head(semantic_feat)
        predicted_offsets, pt_ins_feat = self.instance_head(instance_feat, x)

        return semantic_logits, predicted_offsets, pt_ins_feat, instance_feat

    def test_step(self, batch, batch_idx):
        x = batch
        sem_logits, pred_offsets, pt_ins_feat, raw_features = self(x)
        sem_pred, ins_pred = self.merge_predictions(x, sem_logits,
                                                    pred_offsets, pt_ins_feat)

        if 'RESULTS_DIR' in self.cfg:
            results_dir = self.cfg.RESULTS_DIR
            class_inv_lut = self.evaluator.get_class_inv_lut()
            testing.save_results(sem_pred, ins_pred, results_dir, x, class_inv_lut)

        if 'UPDATE_METRICS' in self.cfg:
            self.evaluator.update(sem_pred, ins_pred, x)

        if 'SAVE_FEATURES' in self.cfg:
            pt_raw_feat = pred.feat_voxel2point(raw_features,x)
            sf.save_features(x, pt_raw_feat, sem_pred, ins_pred, save_preds=False)

        if 'SAVE_VAL_PRED' in self.cfg:
            pt_raw_feat = pred.feat_voxel2point(raw_features,x)
            sf.save_features(x, pt_raw_feat, sem_pred, ins_pred, save_preds=True)

# Modules

class CylinderEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        output_shape = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE
        num_input_features = cfg.MODEL.VOXEL_FEATURES.FEATURE_DIM
        self.nclasses = cfg.DATA_CONFIG.NCLASS
        self.n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        sparse_shape = np.array(output_shape)
        self.sparse_shape = sparse_shape

        self.contextBlock = blocks.ResBlock(num_input_features, init_size, indice_key="context")
        self.downBlock0 = blocks.DownResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down0")
        self.downBlock1 = blocks.DownResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down1")
        self.downBlock2 = blocks.DownResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down2")
        self.downBlock3 = blocks.DownResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down3")

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        x = self.contextBlock(x)
        down0_feat, down0_skip = self.downBlock0(x)
        down1_feat, down1_skip = self.downBlock1(down0_feat)
        down2_feat, down2_skip = self.downBlock2(down1_feat)
        down3_feat, down3_skip = self.downBlock3(down2_feat)

        skips = [down0_skip, down1_skip, down2_skip, down3_skip]

        return down3_feat, skips

class CylinderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.upBlock0 = blocks.UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down3")
        self.upBlock1 = blocks.UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down2")
        self.upBlock2 = blocks.UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down1")
        self.upBlock3 = blocks.UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down0")

        self.decompBlock = blocks.DimDecBlock(2*init_size, 2*init_size, indice_key="decomp")

    def forward(self, x, skips):

        up0 = self.upBlock0(x, skips[3])
        up1 = self.upBlock1(up0, skips[2])
        up2 = self.upBlock2(up1, skips[1])
        up3 = self.upBlock3(up2, skips[0])

        upsampled_feat = self.decompBlock(up3)

        upsampled_feat.features = torch.cat((upsampled_feat.features, up3.features), 1)

        return upsampled_feat, upsampled_feat

class CylinderSemanticHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nclasses = cfg.DATA_CONFIG.NCLASS
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, sem_fea):
        logits = self.logits(sem_fea)
        return logits

class CylinderInstanceHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.pt_fea_dim = 4 * init_size
        self.out_dim = 3 #offset x,y,z

        self.conv1 = blocks.conv3x3(self.pt_fea_dim, self.pt_fea_dim, indice_key='offset_head_conv1')
        self.bn1 = nn.BatchNorm1d(self.pt_fea_dim)
        self.act1 = nn.LeakyReLU()
        self.conv2 = blocks.conv3x3(self.pt_fea_dim, 2 * init_size, indice_key='offset_head_conv2')
        self.bn2 = nn.BatchNorm1d(2 * init_size)
        self.act2 = nn.LeakyReLU()
        self.conv3 = blocks.conv3x3(2 * init_size, init_size, indice_key='offset_head_conv3')
        self.bn3 = nn.BatchNorm1d(init_size)
        self.act3 = nn.LeakyReLU()

        self.offset = nn.Sequential(
            nn.Linear(init_size+3, init_size, bias=True),
            nn.BatchNorm1d(init_size),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(init_size, self.out_dim, bias=True)

    def forward(self, fea, x):
        out = self.conv1(fea)
        out.features = self.act1(self.bn1(out.features))
        out = self.conv2(out)
        out.features = self.act2(self.bn2(out.features))
        out = self.conv3(out)
        out.features = self.act3(self.bn3(out.features))

        grid_ind = x['grid']
        xyz = x['pt_cart_xyz']
        out = out.dense()
        out = out.permute(0, 2, 3, 4, 1)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(out[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        pt_pred_offsets_list = []
        for batch_i, pt_ins_fea in enumerate(pt_ins_fea_list):
            pt_pred_offsets_list.append(self.offset_linear(self.offset(torch.cat([pt_ins_fea,torch.from_numpy(xyz[batch_i]).cuda()],dim=1))))

        return pt_pred_offsets_list, pt_ins_fea_list
