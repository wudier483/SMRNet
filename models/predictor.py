import torch.nn.functional as F
from torch import layer_norm, nn
from torch.nn.parameter import Parameter
import math


from utils import *


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        query = self.query(self.norm(x)).unsqueeze(2)
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = self.norm2(x + y)
        return y


class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, mod_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(mod_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(mod_dim, latent_dim, bias=False)
        self.value = nn.Linear(mod_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, x, xf):
        """
        x:  (64,20,512)
        xf: (64,20,512)
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = self.norm2(x + y)
        return y



class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, node_n=20):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.normal(0, 0.01, (in_features, out_features))) # [512,512]
        self.att = Parameter(torch.normal(0, 0.01, (node_n, node_n)))   # [20,20]
        self.bias = Parameter(torch.zeros(out_features))

    def forward(self, input):
        # input:(batch,node_n,in_features) (64,20,512)
        # return:(batch,node_n,out_features) (64,20,512)
        input = input.to(torch.float32)
        support = torch.matmul(input, self.weight)  # [64,20,512]
        output = torch.matmul(self.att, support)  # [64,20,512]
        return output + self.bias


class FFN(nn.Module):

    def __init__(self, latent_dim, n_pre, dropout):
        super().__init__()
        self.gcn1 = GraphConvolution(latent_dim, latent_dim, n_pre)
        self.gcn2 = GraphConvolution(n_pre, n_pre, latent_dim)
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout(self.activation(self.gcn1(x)))
        y = y.permute(0, 2, 1)
        y = self.dropout(self.activation(self.gcn2(y)))
        y = y.permute(0, 2, 1)
        y = self.linear(y)
        return y



class Layer(nn.Module):
    def __init__(self, latent_dim, n_pre, num_head, dropout):
        super().__init__()
        self.lin1 = FFN(latent_dim, n_pre, dropout)
        self.cross_attention = TemporalCrossAttention(latent_dim, latent_dim, num_head, dropout)
        self.lin2 = FFN(latent_dim, n_pre, dropout)
        self.lin3 = nn.Linear(latent_dim, latent_dim)
        self.lin4 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, mod):
        x1 = self.lin1(x)
        x2 = self.cross_attention(x, mod)
        x2 = self.lin2(x2) + x2
        x3 = x1 + x2
        h = x - self.lin3(x3)
        h_x = self.lin4(x3)
        return h, h_x



class Obse_embed(nn.Module):

    def __init__(self, input_feats, latent_dim, n_pre, num_head, dropout):
        super().__init__()
        self.lin1 = nn.Linear(input_feats, latent_dim)
        self.sequence_embedding = nn.Parameter(torch.randn(n_pre, latent_dim))
        self.gelu = nn.GELU()
        self.att = TemporalSelfAttention(latent_dim, num_head, dropout)
        self.lin2 = FFN(latent_dim, n_pre, dropout)

    def forward(self, x):
        x = self.gelu(self.lin1(x))
        x = self.att(x + self.sequence_embedding.unsqueeze(0))
        x = self.lin2(x) + x
        return x

class Noise_predictor(nn.Module):
    def __init__(self,
                 input_feats,
                 n_pre=20,
                 latent_dim=512,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.3):
        super().__init__()

        self.latent_dim = latent_dim
        self.joint_embed = nn.Linear(input_feats, latent_dim)
        self.obse_embed = Obse_embed(
            input_feats=input_feats,
            latent_dim=latent_dim,
            n_pre=n_pre,
            num_head=num_heads,
            dropout=dropout
        )
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                Layer(
                    latent_dim=latent_dim,
                    n_pre=n_pre,
                    num_head=num_heads,
                    dropout=dropout
                )
            )
        self.out = zero_module(nn.Linear(latent_dim, input_feats))

    def forward(self, x, timesteps, mod):
        """
        x: B, T, D
        """

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim).unsqueeze(1))  # [64,1,512]
        if mod is not None:
            mod_proj = self.obse_embed(mod)
            emb = emb + mod_proj
        h = self.joint_embed(x)
        all_x = 0
        for module in self.temporal_decoder_blocks:
            h, h_x = module(h, emb)
            all_x += h_x
        output = self.out(all_x)
        return output



# t = torch.ones((64)).cuda()
# mod = torch.randn((64, 20, 48)).cuda()
# x = torch.randn((64, 20, 48)).cuda()
#
#
# Predicitor = Noise_predictor(48).cuda()
# params = list(Predicitor.parameters())
# num_params = sum(p.numel() for p in params)
# print(num_params)
# a = Predicitor(x, t, mod)
# print(a.shape)