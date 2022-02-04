"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, pos_labels, sem_labels):
        """Compute loss based on positive and semantic labels to select
           the examples.

        Args:
            features: [n_inst, depth]
            pos_labels: [n_inst, 1] to generate positive examples
            sem_labels: [n_inst, 1] to generate negative examples

        Returns:
            A loss scalar.
        """
        n_inst = features.shape[0]
        pos_mask = torch.eq(pos_labels, pos_labels.T).float()
        sem_mask = torch.eq(sem_labels, sem_labels.T).float()

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(pos_mask.shape[1]).view(-1, 1).to('cuda'),
            0)

        #consider only examples with same semantic class
        logits_mask = logits_mask * sem_mask

        mask = pos_mask * logits_mask

        #don't consider positives in the denominator
        neg_mask = 1-pos_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # exp_logits = torch.exp(logits*neg_mask) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

class AssociationLoss(nn.Module):
    """Quantify associations of current with previous instances based on the
       similarity between the features that depict them
    """
    def __init__(self, life=8, mode='cosine'):
        super(AssociationLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        if mode == 'cosine':
            self.sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        elif mode == 'distance':
            self.sim = torch.dist
        self.life = life

    def forward(self, tracked_instances, ins_feat, ins_ids):
        """Compute Binary Cross Entropy Loss between GT and predicted associations

        Args:
            tracked_instances: dictionary of previous instances of the shape:
                {'id':{'life','feature'}}
            ins_feat: per-instance features tensor of shape [n_ins, depth]
            ins_ids: list of lists of ids of the current instances for each scan

        Returns:
            bce_loss: scalar loss
            tracked_instances: updated dictionary with instances being tracked
        """
        predictions = []
        labels = []
        _bce_loss = []
        i_feat = 0
        for i in range(len(ins_ids)): #for each scan on the batch
            new_instances = {}
            for j in range(len(ins_ids[i])):
                new_instances[ins_ids[i][j]] = {'life': self.life,
                                              'feature': ins_feat[i_feat]}
                i_feat += 1

            if len(tracked_instances) > 0:
                #create features similarity matrix
                pred_assoc = self.get_assoc_matrix(tracked_instances, new_instances, pred=True)
                #create gt association matrix
                gt_assoc = self.get_assoc_matrix(tracked_instances, new_instances)
                assoc_labels = gt_assoc.clone().detach().type(torch.double)

                #Association loss: Binary Cross entropy loss of the similarities
                predictions.append(pred_assoc)
                labels.append(assoc_labels)

                #update tracked_instances with the feature in current frame
                for _id in tracked_instances:
                    if _id in new_instances:
                        tracked_instances[_id]['life'] += 1
                        tracked_instances[_id]['feature'] = new_instances[_id]['feature']
                        del new_instances[_id] #remove already assigned instances

            #Manage new instances
            #add newly created instances to track
            for _id, instance in new_instances.items():
                tracked_instances[_id] = instance

            del new_instances

            # kill instances which are not tracked for a  while
            dont_track_ids = []
            for _id in tracked_instances.keys():
                if tracked_instances[_id]['life'] == 0:
                    dont_track_ids.append(_id)
                else:
                    tracked_instances[_id]['life'] -= 1
            for _id in dont_track_ids:
                del tracked_instances[_id]

        for i in range(len(predictions)):
            _bce_loss.append(self.bce(predictions[i],labels[i]))
        bce_loss = sum(_bce_loss)/len(predictions)

        for _id, instance in tracked_instances.items():
            tracked_instances[_id]['feature'] = tracked_instances[_id]['feature'].detach()

        return bce_loss, tracked_instances

    def get_assoc_matrix(self, previous_instances, current_instances, pred=False):
        p_n = len(previous_instances.keys())
        c_n = len(current_instances.keys())
        matrix = torch.zeros(p_n,c_n).cuda()
        for i, (id1, v1) in enumerate(previous_instances.items()):
            for j, (id2, v2) in enumerate(current_instances.items()):
                if pred:
                    feature_sim = self.sim(v2['feature'], v1['feature'].detach())
                    matrix[i,j] = feature_sim
                else:
                    matrix[i,j] = torch.tensor([int(id1 == id2)])
        return matrix

if __name__ == "__main__":
    pos_labels = torch.tensor([1,2,1,3,2]).unsqueeze(1).to('cuda')
    sem_labels = torch.tensor([1,2,1,2,2]).unsqueeze(1).to('cuda')
    features = torch.tensor([[1.2,1.1],[0.8,0.1],[0.91,0.9],[0.2,1.1],[0.89,1.3]]).to('cuda')
    loss = SupConLoss()
    l = loss(features, pos_labels, sem_labels)
    print("loss",l)
