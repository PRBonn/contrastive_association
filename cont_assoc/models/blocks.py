import MinkowskiEngine as ME
import time
import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from cont_assoc.utils.voxel_features import grp_range_torch

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                     padding=(0, 1, 1), bias=False, indice_key=indice_key)

def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                     padding=(0, 0, 1), bias=False, indice_key=indice_key)

def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                     padding=(0, 1, 0), bias=False, indice_key=indice_key)

def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                     padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                     padding=(1, 0, 1), bias=False, indice_key=indice_key)

def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_dim = cfg.DATA_CONFIG.DATALOADER.DATA_DIM #9
        out_dim = cfg.MODEL.VOXEL_FEATURES.OUT_DIM #64
        point_feature_dim = cfg.MODEL.VOXEL_FEATURES.FEATURE_DIM #16
        self.max_pt = cfg.MODEL.VOXEL_FEATURES.MAX_PT_PER_ENCODE #256

        self.PointNet = nn.Sequential(
            nn.BatchNorm1d(in_dim),

            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_dim)
        )

        self.FeatureCompression = nn.Sequential(
            nn.Linear(out_dim, point_feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        point_features = x['pt_fea']
        voxel_index = x['grid']

        #create tensors
        pt_fea = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in point_features]
        vox_ind = [torch.from_numpy(i).cuda() for i in voxel_index]

        #concatenate everything
        cat_pt_ind = []
        for i_batch in range(len(vox_ind)):
            cat_pt_ind.append(F.pad(vox_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy voxel index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        coordinates = unq.type(torch.int64)

        # subsample pts using random sampling
        grp_ind = grp_range_torch(unq_cnt)[torch.argsort(torch.argsort(unq_inv))] # convert the array that is in the order of grid to the order of cat_pt_feature
        remain_ind = grp_ind < self.max_pt # randomly sample max_pt points inside a grid

        cat_pt_fea = cat_pt_fea[remain_ind,:]
        cat_pt_ind = cat_pt_ind[remain_ind,:]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt,max=self.max_pt)

        # process feature
        processed_cat_pt_fea = self.PointNet(cat_pt_fea)
        #TODO: maybe use pointnet to extract features inside each grid and each grid share the same parameters instead of apply pointnet to global point clouds?
        # This kind of global pointnet is more memory efficient cause otherwise we will have to alloc [480 x 360 x 32 x 64 x C] tensor in order to apply pointnet to each grid

        # choose the max feature for each grid
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0] 

        voxel_features = self.FeatureCompression(pooled_data)

        # stuff pooled data into 4D tensor
        # out_data_dim = [len(pt_fea),self.grid_size[0],self.grid_size[1],self.pt_fea_dim]
        # out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        # out_data[coordinates[:,0],coordinates[:,1],coordinates[:,2],:] = processed_pooled_data
        # out_data = out_data.permute(0,3,1,2)

        del pt_fea, vox_ind

        # return unq, processed_pooled_data
        return coordinates, voxel_features

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super().__init__()
        self.conv_A1 = conv3x1(in_filters, out_filters, indice_key=indice_key+"bef")
        self.act_A1 = nn.LeakyReLU()
        self.bn_A1 = nn.BatchNorm1d(out_filters)

        self.conv_A2 = conv1x3(out_filters, out_filters, indice_key=indice_key+"bef")
        self.act_A2 = nn.LeakyReLU()
        self.bn_A2 = nn.BatchNorm1d(out_filters)

        self.conv_B1 = conv1x3(in_filters, out_filters, indice_key=indice_key+"bef")
        self.act_B1 = nn.LeakyReLU()
        self.bn_B1 = nn.BatchNorm1d(out_filters)

        self.conv_B2 = conv3x1(out_filters, out_filters, indice_key=indice_key+"bef")
        self.act_B2 = nn.LeakyReLU()
        self.bn_B2 = nn.BatchNorm1d(out_filters)

    def forward(self, x):
        res_B = self.conv_B1(x)
        res_B.features = self.act_B1(res_B.features)
        res_B.features = self.bn_B1(res_B.features)

        res_B = self.conv_B2(res_B)
        res_B.features = self.act_B2(res_B.features)
        res_B.features = self.bn_B2(res_B.features)

        res_A = self.conv_A1(x)
        res_A.features = self.act_A1(res_A.features)
        res_A.features = self.bn_A1(res_A.features)

        res_A = self.conv_A2(res_A)
        res_A.features = self.act_A2(res_A.features)
        res_A.features = self.bn_A2(res_A.features)

        res_B.features = res_B.features + res_A.features

        return res_B

class DownResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, is_dropout=True, height_pooling=False, indice_key=None):
        super().__init__()
        self.is_dropout = is_dropout

        self.conv_A1 = conv3x1(in_filters, out_filters, indice_key=indice_key+"bef")
        self.act_A1 = nn.LeakyReLU()
        self.bn_A1 = nn.BatchNorm1d(out_filters)

        self.conv_A2 = conv1x3(out_filters, out_filters, indice_key=indice_key+"bef")
        self.act_A2 = nn.LeakyReLU()
        self.bn_A2 = nn.BatchNorm1d(out_filters)

        self.conv_B1 = conv1x3(in_filters, out_filters, indice_key=indice_key+"bef")
        self.act_B1 = nn.LeakyReLU()
        self.bn_B1 = nn.BatchNorm1d(out_filters)

        self.conv_B2 = conv3x1(out_filters, out_filters, indice_key=indice_key+"bef")
        self.act_B2 = nn.LeakyReLU()
        self.bn_B2 = nn.BatchNorm1d(out_filters)

        # self.dropout = nn.Dropout3d(p=dropout_rate)
        if height_pooling:
            _stride = 2 #pooling in all dimensions
        else:
            _stride = (2,2,1) #not pooling in z
        # self.pool = spconv.SparseMaxPool3d(kernel_size=2, stride=_stride)
        self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3,
            stride=_stride, padding=1, indice_key=indice_key, bias=False)

    def forward(self, x):
        res_A = self.conv_A1(x)
        res_A.features = self.act_A1(res_A.features)
        res_A.features = self.bn_A1(res_A.features)

        res_A = self.conv_A2(res_A)
        res_A.features = self.act_A2(res_A.features)
        res_A.features = self.bn_A2(res_A.features)

        res_B = self.conv_B1(x)
        res_B.features = self.act_B1(res_B.features)
        res_B.features = self.bn_B1(res_B.features)

        res_B = self.conv_B2(res_B)
        res_B.features = self.act_B2(res_B.features)
        res_B.features = self.bn_B2(res_B.features)

        res_B.features = res_B.features + res_A.features

        # if self.is_dropout:
        #     downSampled = self.dropout(B.features)
        # else:
        #     downSampled = B
        downSampled = self.pool(res_B)

        return downSampled, res_B

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super().__init__()

        # self.is_dropout = drop_out

        self.conv1 = conv3x3(in_filters, out_filters, indice_key=indice_key+"new_up")
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        # self.dropout1 = nn.Dropout3d(p=dropout_rate)

        self.upsample = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key, bias=False)

        # self.dropout2 = nn.Dropout3d(p=dropout_rate)

        self.conv2 = conv1x3(out_filters, out_filters,  indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters,  indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)

        self.conv4 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(out_filters)

        # self.dropout4 = nn.Dropout3d(p=dropout_rate)

    def forward(self, x, skip):

        up = self.conv1(x)
        up.features = self.act1(up.features)
        up.features = self.bn1(up.features)

        ## upsample
        up = self.upsample(up)
        # up = F.interpolate(up, size=skip.size()[2:], mode='trilinear', align_corners=True)

        # if self.is_dropout:
        #     up = self.dropout1(up)
        up.features = up.features + skip.features
        # if self.is_dropout:
        #     up = self.dropout2(up)

        up = self.conv2(up)
        up.features = self.act2(up.features)
        up.features = self.bn2(up.features)

        up = self.conv3(up)
        up.features = self.act3(up.features)
        up.features = self.bn3(up.features)

        up = self.conv4(up)
        up.features = self.act4(up.features)
        up.features = self.bn4(up.features)

        # if self.drop_out:
        #     up = self.dropout3(up)

        return up


class DimDecBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super().__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key+"bef")
        self.bn1 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key+"bef")
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.act2 = nn.Sigmoid()

        self.conv3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key+"bef")
        self.bn3 = nn.BatchNorm1d(out_filters)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out1.features = self.bn1(out1.features)
        out1.features = self.act1(out1.features)


        out2 = self.conv2(x)
        out2.features = self.bn2(out2.features)
        out2.features = self.act2(out2.features)


        out3 = self.conv3(x)
        out3.features = self.bn3(out3.features)
        out3.features = self.act3(out3.features)

        out1.features = out1.features + out2.features + out3.features

        out1.features = out1.features * x.features

        return out1

class LinRel(nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        self.layer = nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.LeakyReLU()
                        )

    def forward(self, x):
        return self.layer(x)

### Blocks with ME

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

def split_sparse(sparse, n_ins):
    #n_ins = [7,9]
    all_batched = []
    cont = 0
    coords, feats = sparse.decomposed_coordinates_and_features
    for i in range(len(n_ins)):
        batched = []
        for j in range(n_ins[i]):
            single_sparse = ME.SparseTensor(features=feats[cont], coordinates=coords[cont])
            batched.append(single_sparse)
            cont += 1
        all_batched.append(batched)
    return all_batched
