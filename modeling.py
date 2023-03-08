import torch
import torch.nn as nn
import torch.nn.functional as F
from global_configs import *

class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()
        print(
            "Initializing MAG with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )

        self.W_hv = nn.Linear(TEXT_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(TEXT_DIM + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.W_a = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.dim_t, self.dim_v, self.dim_a = TEXT_DIM, VISUAL_DIM, ACOUSTIC_DIM
        self.proj_v = nn.Linear(self.dim_v, self.dim_t)
        self.proj_a = nn.Linear(self.dim_a, self.dim_t)

        self.pa_gate = nn.parameter.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        # 1. 先将feature卷积至相同的hidden_dim.
        visual = self.proj_v(visual)
        acoustic = self.proj_a(acoustic)

        # 2. PA: pm assign.
        prob = torch.softmax(self.pa_gate, dim=-1)
        # print(prob)
        _pm = torch.argmax(prob)
        _pm = 2
        if _pm == 0:
            ...
        elif _pm == 1:
            tmp = text_embedding
            text_embedding = visual
            visual = tmp
        elif _pm == 2:
            tmp = text_embedding
            text_embedding = acoustic
            acoustic = tmp
        # weight matrices [W] & non-linear activation [R()]
        # input(f'text: {text_embedding.shape}; visual: {visual.shape}; audio : {acoustic.shape}')
        # OG: visual: [50, 48, 768], [50, 48, 47], [50, 48, 74]  # [len, bsz, dim]
        # new: [50, 48, 768].

        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)  # [50, 48, 768]
        # input(f'h_m: {h_m.shape}')

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output
