import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MatchLoss(nn.Module):
    """
    Contrastive loss
    Takes match predictions of two samples and puts a sigmoid loss on them
    """

    def __init__(self, weighted_loss=False):
        super(MatchLoss, self).__init__()
        self.weighted_loss = weighted_loss
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.eps = 1e-9

    def forward(self, match_logits, target):
        '''
            matches: logits 
            target : binary values
        '''
        if not self.weighted_loss :
            loss = self.criterion(match_logits, target)
        else :
            pos_ix = np.where(target.cpu().numpy() == 1)[0]
            num_pos = len(pos_ix)

            neg_ix = np.where(target.cpu().numpy() == 0)[0]
            num_neg = len(neg_ix)

            weights = np.ones((num_pos + num_neg))
            weights[pos_ix] = num_neg/num_pos
            weights = weights/np.sum(weights)
            weights = torch.Tensor(weights).cuda()
            criterion = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')
            loss = criterion(match_logits, target)
        return loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class MarginLoss(nn.Module) :

    def __init__(self, margin, beta, triplet_selector) :
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.beta = beta
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target) :

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = torch.sqrt((embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1) + 1e-6)
        an_distances = torch.sqrt((embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1) + 1e-6)

        pos_loss = F.relu(ap_distances - self.beta + self.margin)
        neg_loss = F.relu(self.beta - an_distances + self.margin)
        pair_cnt = torch.sum((pos_loss > 0) + (neg_loss > 0))

        assert(pair_cnt.item() > 0)
        loss = torch.sum(pos_loss + neg_loss).div(pair_cnt)

        return loss, len(triplets)