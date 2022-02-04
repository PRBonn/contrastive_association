import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as schedul
from pytorch_lightning.core.lightning import LightningModule
import cont_assoc.models.blocks as blocks
import cont_assoc.utils.contrastive as cont
from cont_assoc.utils.assoc_module import AssociationModule
from cont_assoc.utils.evaluate_4dpanoptic import PanopticKitti4DEvaluator
import cont_assoc.utils.tracking as t
from cont_assoc.models.loss_contrastive import SupConLoss, AssociationLoss

class ContrastiveTracking(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = SparseEncoder(cfg)
        self.cont_loss = SupConLoss(temperature=cfg.TRAIN.CONTRASTIVE_TEMP)

        self.evaluator4D = PanopticKitti4DEvaluator(cfg=cfg)

        feat_size = cfg.DATA_CONFIG.DATALOADER.DATA_DIM
        self.pos_enc = cont.PositionalEncoder(max_freq=100000,
                                              feat_size=feat_size,
                                              dimensionality=3)

        weights = cfg.TRACKING.ASSOCIATION_WEIGHTS
        thresholds = cfg.TRACKING.ASSOCIATION_THRESHOLDS
        use_poses = cfg.MODEL.USE_POSES
        self.AssocModule = AssociationModule(weights, thresholds, self.encoder,
                                             self.pos_enc, use_poses)

    def getLoss(self, x, features):
        loss = {}
        sem_labels = [torch.from_numpy(i).type(torch.LongTensor).cuda()
                      for i in x['sem_label']]
        sem_labels = (torch.cat([i for i in sem_labels])).unsqueeze(1) #single tensor
        pos_labels = [torch.from_numpy(i).type(torch.LongTensor).cuda()
                      for i in x['pos_label']]
        pos_labels = (torch.cat([i for i in pos_labels])).unsqueeze(1) #single tensor
        norm_features = F.normalize(features)
        contrastive_loss = self.cont_loss(norm_features, pos_labels, sem_labels)
        loss['cont'] = contrastive_loss
        return loss

    def sparse_tensor(self, pt_coors,pt_features):
        for i in range(len(pt_features)):
            for j in range(len(pt_features[i])):
                pos_encoding = self.pos_enc(pt_coors[i][j])
                pt_features[i][j] = pt_features[i][j] + pos_encoding
        #create sparse tensor
        all_feat = [item for sublist in pt_features for item in sublist]
        all_coors = [item for sublist in pt_coors for item in sublist]
        c_, f_ = ME.utils.sparse_collate(all_coors, all_feat, dtype=torch.float32)
        sparse = ME.SparseTensor(features=f_, coordinates=c_.int(),device='cuda')
        return sparse

    def forward(self, x):
        coors = x['pt_coors']
        sparse = self.sparse_tensor(coors, x['pt_features'])
        ins_features = self.encoder(sparse)
        return ins_features

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                self.parameters()), lr=self.cfg.TRAIN.LR)

        eta_min=self.cfg.TRAIN.LR/self.cfg.TRAIN.SCHEDULER.DIV_FACTOR
        scheduler = schedul.CosineAnnealingLR(optimizer,
                                              self.cfg.TRAIN.MAX_EPOCH,
                                              eta_min=eta_min)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch
        instance_features = self(x)
        loss = self.getLoss(x, instance_features)
        self.log('train/cont_loss', loss['cont'])
        torch.cuda.empty_cache()

        return loss['cont']

    def validation_step(self, batch, batch_idx):
        x = batch
        instance_features = self(x)

        if 'ONLY_SEQ' in self.cfg.TRAIN.keys():
            torch.cuda.empty_cache()
            return

        #load predictions for the whole scan (merge_predictions)
        ins_pred = x['pt_ins_pred']
        sem_pred = x['pt_sem_pred']
        ins_ids = x['id']
        n_instances = [len(item) for item in x['id']]
        ins_feat = torch.split(instance_features, n_instances)
        batched_ins_feat = torch.split(instance_features, n_instances)
        points = x['pt_coors']
        features = x['pt_features']
        poses = x['pose']

        ins_pread = self.AssocModule.associate(ins_pred, ins_feat, points,
                                               features, poses, ins_ids)

        self.evaluator4D.update(sem_pred, ins_pred, x)

        torch.cuda.empty_cache()

        return

    def validation_epoch_end(self, outputs):
        self.evaluator4D.calculate_metrics()
        AQ = self.evaluator4D.get_mean_aq()
        self.log('AQ',AQ)

        self.AssocModule.clear()
        self.evaluator4D.clear()

    def test_step(self, batch, batch_idx):
        x = batch
        instance_features = self(x)

        if 'ONLY_SEQ' in self.cfg.TRAIN.keys():
            torch.cuda.empty_cache()
            return
        ins_pred = x['pt_ins_pred']
        sem_pred = x['pt_sem_pred']
        ins_ids = x['id']
        n_instances = [len(item) for item in x['id']]
        ins_feat = torch.split(instance_features, n_instances)
        batched_ins_feat = torch.split(instance_features, n_instances)
        points = x['pt_coors']
        features = x['pt_features']
        poses = x['pose']

        ins_pread = self.AssocModule.associate(ins_pred, ins_feat, points,
                                               features, poses, ins_ids)

        self.evaluator4D.update(sem_pred, ins_pred, x)

        torch.cuda.empty_cache()

# Modules
class SparseEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.DATA_CONFIG.DATALOADER.DATA_DIM #128
        channels = [x * input_dim for x in cfg.MODEL.ENCODER.CHANNELS] #128, 128, 256, 512
        kernel_size = 3

        self.conv1 = SparseConvBlock(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = SparseConvBlock(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv3 = SparseConvBlock(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
        )

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            SparseLinearBlock(channels[-1], 2*channels[-1]),
            ME.MinkowskiDropout(),
            SparseLinearBlock(2*channels[-1], channels[-1]),
            ME.MinkowskiLinear(channels[-1], channels[-1], bias=True),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.global_avg_pool(y)
        return self.final(y).F

class SparseLinearBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
                        ME.MinkowskiLinear(in_channel, out_channel, bias=False),
                        ME.MinkowskiBatchNorm(out_channel),
                        ME.MinkowskiLeakyReLU(),
                    )

    def forward(self, x):
        return self.layer(x)

class SparseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dimension=3):
        super().__init__()
        self.layer =  nn.Sequential(
                        ME.MinkowskiConvolution(
                            in_channel,
                            out_channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            dimension=dimension),
                        ME.MinkowskiBatchNorm(out_channel),
                        ME.MinkowskiLeakyReLU(),
                    )

    def forward(self, x):
        return self.layer(x)
