import torch
import pandas as pd
import numpy as np
import torch.nn as nn 
import sklearn
from sklearn.neighbors import NearestNeighbors as sknn
from itertools import combinations
from maskrcnn_benchmark.modeling.retrieval.siamese_utils import * 
from maskrcnn_benchmark.modeling.retrieval.siamese_losses import * 

class EmbeddingNet(nn.Module) :
    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(EmbeddingNet, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.ReLU()
        )

        self.embedding = nn.Sequential(
            nn.Linear(7*7*256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )

        self.sampler = None 
        if cfg.RETRIEVAL.EMBEDDING.SAMPLER == 'semihard' :
            self.sampler = SemihardNegativeTripletSelector(cfg.RETRIEVAL.EMBEDDING.MARGIN)
        self.criterion = OnlineTripletLoss(cfg.RETRIEVAL.EMBEDDING.MARGIN, self.sampler)

    def compute_TA_rate(self, embedding, ids) :
        '''
            true acceptance rate
        '''
        embedding = embedding.detach().cpu().numpy()
        cluster = sknn().fit(embedding)

        ids = ids.numpy()
        id_counts = pd.Series(ids).value_counts() 
        id_matches = np.array(id_counts[(id_counts > 1)].index)

        retrieved = 0
        for id in id_matches :
            ids_ixes = np.where(id==ids)[0]
            query_ix = ids_ixes[0]
            query_feature = embedding[query_ix, :].reshape(1, -1)
            dist, nbr_ixes = cluster.kneighbors(query_feature, n_neighbors=2)
            if nbr_ixes[0][1] == ids_ixes[1] :
                retrieved += 1
        TA_rate = retrieved/len(id_matches)
        return TA_rate

    def forward(self, features, ids) :
        embedding_input = self.downsample(features)
        embedding_input = embedding_input.view(-1, 7*7*256) # flatten
        embedding = self.embedding(embedding_input)

        l2norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        embedding_l2norm = embedding.div(l2norm.expand_as(embedding))
        loss, _ = self.criterion(embedding_l2norm, ids)

        TA_rate = self.compute_TA_rate(embedding_l2norm, ids)
        return loss, TA_rate

class MatchNet(nn.Module) :
    def __init__(self, cfg) :
        super(MatchNet, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.ReLU()
        )

        self.embedding = nn.Sequential(
            nn.Linear(7*7*256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )

        self.match = nn.Linear(256, 1)

        self.criterion = MatchLoss(weighted_loss=True)

    def forward(self, features, ids) :

        embedding_input = self.downsample(features)
        embedding_input = embedding_input.view(-1, 7*7*256) # flatten
        embedding = self.embedding(embedding_input)

        l2norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        embedding_l2norm = embedding.div(l2norm.expand_as(embedding))

        # compute pairwise matches 
        ixes = np.arange(features.shape[0]).tolist()
        pairwise_ixes = list(combinations(ixes, 2)) 

        predictions = []
        for pair in pairwise_ixes :
            e1, e2 = embedding_l2norm[pair[0]], embedding_l2norm[pair[1]]
            match = self.match(torch.sqrt(torch.pow(e1-e2, 2)))
            predictions.append(match)
        match_logits = torch.cat(predictions)

        pairwise_ids = ids[[pairwise_ixes]] 
        match_targets = torch.sub(pairwise_ids[:, 0], pairwise_ids[:, 1])
        match_targets[np.where(match_targets.numpy() < 0)[0]] = 1
        match_targets = 1-match_targets 
        match_targets = match_targets.cuda()

        loss = self.criterion(match_logits, match_targets)

        return loss

def build_retrieval_model(cfg) :
    if cfg.RETRIEVAL.METHOD == 'match' :
        model = MatchNet(cfg)
    else :
        model = EmbeddingNet(cfg)
    return model

