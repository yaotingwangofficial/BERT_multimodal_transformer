import torch
import torch.nn as nn
import torch.nn.functional as F
from global_configs import *
from modules.mult.transformer import TransformerEncoder


class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()
        print(
            "Initializing MAG with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )
        self.dim_t, self.dim_v, self.dim_a = TEXT_DIM, VISUAL_DIM, ACOUSTIC_DIM
        self.dim_tmp = 40

        self.W_hv = nn.Linear(self.dim_tmp + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(self.dim_tmp + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(self.dim_tmp, TEXT_DIM)
        self.W_a = nn.Linear(self.dim_tmp, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.kernels = [1, 1, 1]  # kernel for [T, A, V] respectively.

        # self.proj_v = nn.Linear(self.dim_v, self.dim_t)
        # self.proj_a = nn.Linear(self.dim_a, self.dim_t)
        _bias = True
        self.proj_t = nn.Conv1d(self.dim_t, self.dim_tmp, kernel_size=self.kernels[1], padding=0, bias=_bias)
        self.proj_v = nn.Conv1d(self.dim_v, self.dim_tmp, kernel_size=self.kernels[1], padding=0, bias=_bias)
        self.proj_a = nn.Conv1d(self.dim_a, self.dim_tmp, kernel_size=self.kernels[2], padding=0, bias=_bias)
        self.dim_out = 1
        self.output_concat = nn.Linear(self.dim_tmp*3, self.dim_out)

        # self.proj_t_2 = nn.Linear(self.dim_t, self.dim_tmp)

        """ 2. GRU """
        num_layers = 3  # TODO: GRU layers?
        # BF = False  # is_batch_first?
        BD = False  # is_bidirectional? # 设置成True, 后面会有 40*2=80, 还是应该取后面[:40]?
        # seq_len 不包含在内
        in_size = 768
        out_size = 768
        BIAS = False  # is_bias_open?
        self.gru_a = nn.GRU(in_size, out_size, num_layers, bidirectional=BD, bias=BIAS)  #
        self.gru_v = nn.GRU(in_size, out_size, num_layers, dropout=0, bidirectional=BD, bias=BIAS)  # default-setting

        self.pa_gate = nn.parameter.Parameter(torch.ones(3), requires_grad=True)

        # self.net_prim = self.get_network('mem_prim')
        self.attn_dropout = 0.1
        self.layers = 5
        self.heads = 2
        self.embed_dropout = 0.1
        self.attn_mask = None
        self.net_prim_with_aux1 = self.get_network('p_a1')
        self.net_prim_with_aux2 = self.get_network('p_a2')

    def get_network(self, net_type='mem_t', layers=-1):
        if net_type in ['mem_t', 'mem_prim']:
            embed_dim, attn_dropout = 1 * self.dim_tmp, self.attn_dropout
        # elif net_type in ['final_concat']:
        #     embed_dim, attn_dropout = 2 * self.dim_t, self.attn_dropout
        elif net_type == 'mem_a':
            embed_dim, attn_dropout = 1 * self.dim_tmp, self.attn_dropout
        elif net_type == 'mem_v':
            embed_dim, attn_dropout = 1 * self.dim_tmp, self.attn_dropout
        elif net_type in ['p_a1', 'p_a2', 'p_a']:
            embed_dim, attn_dropout = 1 * self.dim_tmp, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        # mult transformer
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  # relu_dropout=self.relu_dropout,
                                  # res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        # 1. 先将feature卷积至相同的hidden_dim.
        text_embedding_tmp = self.proj_t(text_embedding.transpose(1, 2)).permute(2, 0, 1)
        text_embedding = text_embedding.transpose(1, 2).permute(2, 0, 1)
        visual = self.proj_v(visual.transpose(1, 2)).permute(2, 0, 1)
        acoustic = self.proj_a(acoustic.transpose(1, 2)).permute(2, 0, 1)

        # 取出最后一层, 用于预测uni-modal的结果.
        # input(f'visual: {visual.shape}')  # [50, 48, 40]

        # 2. GRU
        # text_embedding, f_t = self.gru_l(text_embedding)
        # acoustic, f_a = self.gru_a(acoustic)
        # visual, f_v = self.gru_v(visual)

        # 3. 回秩
        # text_embedding = text_embedding.permute(1, 0, 2)
        # acoustic = acoustic.permute(1, 0, 2)
        # visual = visual.permute(1, 0, 2)

        # print(text_embedding.shape)

        # 2. PA: pm assign.
        prob = torch.softmax(self.pa_gate, dim=-1)
        # print(prob)
        _pm = torch.argmax(prob)
        _pm = 0
        if _pm == 0:
            ...
        elif _pm == 1:
            tmp = text_embedding_tmp
            text_embedding = visual
            visual = tmp
        elif _pm == 2:
            tmp = text_embedding_tmp
            text_embedding = acoustic
            acoustic = tmp
        # weight matrices [W] & non-linear activation [R()]
        # input(f'text: {text_embedding.shape}; visual: {visual.shape}; audio : {acoustic.shape}')
        # OG: visual: [50, 48, 768], [50, 48, 47], [50, 48, 74]  # [len, bsz, dim]
        # new: [50, 48, 768].

        visual = self.net_prim_with_aux1(text_embedding_tmp, visual, visual)
        acoustic = self.net_prim_with_aux2(text_embedding_tmp, acoustic, acoustic)
        # print(visual.shape, text_embedding_tmp.shape, text_embedding.shape)
        # input('---')

        weight_v = F.leaky_relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.leaky_relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)  # [50, 48, 768]
        # input(f'h_m: {h_m.shape}, {text_embedding.shape}')

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m
        # input(f'alpha.shape: {alpha.shape}')  # [50, 32, 1]

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        # input(f'embd_shape: {embedding_output.shape}')  # input(f'embd: {embedding_output.shape}')  # OG: [48, 50, 768]

        x_t = text_embedding_tmp[-1]
        x_v = visual[-1]
        x_a = acoustic[-1]
        return embedding_output.transpose(0, 1), [x_t, x_v, x_a]

