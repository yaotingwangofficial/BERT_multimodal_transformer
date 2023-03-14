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
        self.dim_gate = 256
        self.proj_v = nn.Linear(self.dim_v, self.dim_t)
        self.proj_a = nn.Linear(self.dim_a, self.dim_t)

        # self.pa_gate = nn.parameter.Parameter(torch.ones(3), requires_grad=True)
        self.n_mods = 3
        self.net_gate = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.dim_t, self.dim_gate),
                nn.LeakyReLU(),
                nn.Linear(self.dim_gate, 1),
                nn.Sigmoid()
            ) for _i in range(self.n_mods)]
        )
        self.g_mean = None
        assert id(self.net_gate[0]) != id(self.net_gate[1])

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        # 1. 先将feature卷积至相同的hidden_dim.
        visual = self.proj_v(visual)
        acoustic = self.proj_a(acoustic)

        # 2. PA: pm assign.
        last_h_t = text_embedding[-1]
        last_h_v = visual[-1]
        last_h_a = acoustic[-1]
        # print(last_h_t.shape, last_h_v.shape, last_h_a.shape)  # [50, 768]
        # input()
        g_t, g_v, g_a = [self.net_gate[_i](_m) for _i, _m in enumerate([last_h_t, last_h_v, last_h_a])]
        g_t_mean = torch.mean(g_t, dim=0)
        g_v_mean = torch.mean(g_v, dim=0)
        g_a_mean = torch.mean(g_a, dim=0)
        self.g_mean = torch.stack([g_t_mean[0], g_a_mean[0], g_v_mean[0]])

        _pm = torch.argmax(self.g_mean)
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

        # TODO: 这里是直接concat, 想想如何用transformer, 进行信息提取... (析取).
        # 这里是拼接text+other, 然后只输出dim=[一个模态]的activation, 作为text-relative 的 其余模态的 "参考权重".
        weight_v = F.leaky_relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.leaky_relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        # summation fusion.
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)  # [50, 48, 768]
        # input(f'h_m: {h_m.shape}')

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        # print(hm_norm)
        # input()

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        # print(hm_norm)
        # input("---")

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        x_t = text_embedding[-1]
        x_v = visual[-1]
        x_a = acoustic[-1]
        return embedding_output, [x_t, x_v, x_a]
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
        self.dim_gate = 256
        self.proj_v = nn.Linear(self.dim_v, self.dim_t)
        self.proj_a = nn.Linear(self.dim_a, self.dim_t)

        # self.pa_gate = nn.parameter.Parameter(torch.ones(3), requires_grad=True)
        self.n_mods = 3
        self.net_gate = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.dim_t, self.dim_gate),
                nn.LeakyReLU(),
                nn.Linear(self.dim_gate, 1),
                nn.Sigmoid()
            ) for _i in range(self.n_mods)]
        )
        self.g_mean = None
        assert id(self.net_gate[0]) != id(self.net_gate[1])

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        # 1. 先将feature卷积至相同的hidden_dim.
        visual = self.proj_v(visual)
        acoustic = self.proj_a(acoustic)

        # 2. PA: pm assign.
        last_h_t = text_embedding[-1]
        last_h_v = visual[-1]
        last_h_a = acoustic[-1]
        # print(last_h_t.shape, last_h_v.shape, last_h_a.shape)  # [50, 768]
        # input()
        g_t, g_v, g_a = [self.net_gate[_i](_m) for _i, _m in enumerate([last_h_t, last_h_v, last_h_a])]
        g_t_mean = torch.mean(g_t, dim=0)
        g_v_mean = torch.mean(g_v, dim=0)
        g_a_mean = torch.mean(g_a, dim=0)
        self.g_mean = torch.stack([g_t_mean[0], g_a_mean[0], g_v_mean[0]])

        _pm = torch.argmax(self.g_mean)
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

        # TODO: 这里是直接concat, 想想如何用transformer, 进行信息提取... (析取).
        # 这里是拼接text+other, 然后只输出dim=[一个模态]的activation, 作为text-relative 的 其余模态的 "参考权重".
        weight_v = F.leaky_relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.leaky_relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        # summation fusion.
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)  # [50, 48, 768]
        # input(f'h_m: {h_m.shape}')

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        # print(hm_norm)
        # input()

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        # print(hm_norm)
        # input("---")

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        x_t = text_embedding[-1]
        x_v = visual[-1]
        x_a = acoustic[-1]
        return embedding_output, [x_t, x_v, x_a]
