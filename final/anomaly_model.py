import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model):
        super(AnomalyAttention, self).__init__()
        self.N = N
        self.d_model = d_model

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False)  # For sigma in prior association

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        sigma = torch.clamp(self.Ws(x), min=1e-3)  # For sigma in prior association
        # Prior association
        P = self.prior_association(sigma)

        # Series association
        S = (Q @ K.T) / math.sqrt(self.d_model)
        S = (S - torch.mean(S, dim=-1)) / torch.std(S, dim=-1)
        S = torch.softmax(S, dim=-1)

        Z = S @ V

        return Z, P, S

    @staticmethod
    def prior_association(sigma):
        N = sigma.shape[0]
        p = torch.from_numpy(np.abs(np.indices((N, N))[0] - np.indices((N, N))[1]))
        gaussian = torch.exp(-0.5 * (p / sigma).pow(2)) / torch.sqrt(2 * torch.pi * sigma)
        prior_ass = gaussian / gaussian.sum(axis=1, keepdim=True)
        return prior_ass


class AnomalyTransformer(nn.Module):
    def __init__(self, N, d_model, hidden_dim, lambda_=0.1):
        super(AnomalyTransformer, self).__init__()
        self.N = N
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.lambda_ = lambda_

        self.attention_layers = AnomalyAttention(N, d_model)
        self.hiddenlayer = nn.Linear(d_model, hidden_dim)
        self.outputLayer = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        Z, P, S = self.attention_layers(x)
        hidden = torch.relu(self.hiddenlayer(Z))
        x_hat = self.outputLayer(hidden)

        return x_hat, P, S

    def association_discrepancy(self, P_list, S_list):
        ass_diss = (1 / len(P_list)) * torch.tensor(
            [
                self.layer_association_discrepancy(P, S)
                for P, S in zip(P_list, S_list)
            ]
        )
        return ass_diss

    def layer_association_discrepancy(self, Pl, Sl):
        epsilon = 1e-10
        kl_div_sum = F.kl_div(Pl + epsilon, Sl + epsilon, reduction="sum") + F.kl_div(Sl + epsilon, Pl + epsilon, reduction="sum")
        return kl_div_sum

    def loss_function(self, x_hat, x, P_list, S_list, lambda_):
        frob_norm = torch.linalg.norm(x_hat - x, ord="fro")
        assoc_disc = self.association_discrepancy(P_list, S_list)
        kl_div = torch.norm(assoc_disc, p=1)
        return frob_norm + lambda_ * kl_div

    # min_loss function to calculate the loss for the min-discrepancy term
    def min_loss(self, x_hat, x, P_list, S_list):
        p_list_detach = [P.detach() for P in P_list]
        return self.loss_function(x_hat, x, p_list_detach, S_list, -self.lambda_)

    # max_loss function to calculate the loss for the max-discrepancy term
    def max_loss(self, x_hat, x, P_list, S_list):
        s_list_detach = [S.detach() for S in S_list]
        return self.loss_function(x_hat, x, P_list, s_list_detach, self.lambda_)

    def anomaly_score(self, x):
        x_hat, P_list, S_list = self(x)
        
        assoc_dis = self.association_discrepancy(P_list, S_list)

        ad = F.softmax(-assoc_dis, dim=0)
        print(f"ad: {ad.shape}")
        reconstruction_error = torch.linalg.norm((x - x_hat)**2, dim=1)
        print(f"re: {reconstruction_error.shape}")
        print(f"x_hat: {x_hat.shape}")
        print(f"x: {x.shape}")
        return ad * reconstruction_error
