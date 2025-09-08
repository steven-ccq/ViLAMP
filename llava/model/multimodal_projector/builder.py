import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import re

from .pooler_projector import PoolerProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")

class AveragePoolingProjector():
    def __init__(self, patch_num=729):
        self.patch_num = patch_num
    def forward(self, embs):
        pooled_embs = []
        for emb in embs:
            assert len(emb.shape) == 2 and emb.shape[0] == self.patch_num
            pooled_embs.append(emb.sum(0)/emb.shape[0])
        return pooled_embs

class KeyFrameProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2):
        return self.alpha * x1 + (1-self.alpha) * x2

# class SigPooling(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.linear_iq = nn.Linear(config.hidden_size, config.hidden_size)
#         self.linear_ik = nn.Linear(config.hidden_size, config.hidden_size)
#         self.linear_q = nn.Linear(config.composer_proj_dim, config.hidden_size)
#         self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tau = nn.Parameter(torch.tensor(0.01))

#     def forward(self, img_embs, key_frame_expanded, query_expanded):
#         iq = self.linear_iq(img_embs)
#         ik = self.linear_ik(img_embs)
#         q = self.linear_q(query_expanded)
#         k = self.linear_k(key_frame_expanded)
#         score_i2q = F.cosine_similarity(q, iq, dim=-1)
#         score_i2k = F.cosine_similarity(k, ik, dim=-1)
#         score = (score_i2q - score_i2k) / self.tau
#         mask = F.softmax(score, dim=-1).unsqueeze(-1)
#         img_embs = mask * img_embs
#         img_embs = img_embs.sum(1)

#         return img_embs

class SigPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_iq = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_ik = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_q = nn.Linear(config.composer_proj_dim, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        init.normal_(self.linear_iq.weight, mean=0.0, std=0.02)
        init.normal_(self.linear_ik.weight, mean=0.0, std=0.02)
        init.normal_(self.linear_q.weight, mean=0.0, std=0.02)
        init.normal_(self.linear_k.weight, mean=0.0, std=0.02)
        self.tau = nn.Parameter(torch.tensor(0.01))

    def forward(self, img_embs, key_frame_expanded, query_expanded, max_dense_length=60):
        iq = self.linear_iq(img_embs)
        ik = self.linear_ik(img_embs)
        q = self.linear_q(query_expanded)
        k = self.linear_k(key_frame_expanded)
        if img_embs.size()[0] > max_dense_length:
            iq_chunks = list(torch.unbind(iq, dim=0))
            q_chunks = list(torch.unbind(q, dim=0))
            score_i2q = []
            for idx in range(0, len(iq_chunks), max_dense_length):
                cur_iq = torch.stack(iq_chunks[idx: min(idx+max_dense_length, len(iq_chunks))], dim=0).cpu()
                cur_q = torch.stack(q_chunks[idx: min(idx+max_dense_length, len(q_chunks))], dim=0).cpu()
                cur_score_i2q = F.cosine_similarity(cur_q, cur_iq, dim=-1)
                score_i2q.append(cur_score_i2q)
                del cur_iq, cur_q, cur_score_i2q
            score_i2q = torch.cat(score_i2q, dim=0).to(img_embs.device)

            ik_chunks = list(torch.unbind(ik, dim=0))
            k_chunks = list(torch.unbind(k, dim=0))
            score_i2k = []
            for idx in range(0, len(ik_chunks), max_dense_length):
                cur_ik = torch.stack(ik_chunks[idx: min(idx+max_dense_length, len(ik_chunks))], dim=0).cpu()
                cur_k = torch.stack(k_chunks[idx: min(idx+max_dense_length, len(k_chunks))], dim=0).cpu()
                cur_score_i2k = F.cosine_similarity(cur_k, cur_ik, dim=-1)
                score_i2k.append(cur_score_i2k)
                del cur_k, cur_ik
            score_i2k = torch.cat(score_i2k, dim=0).to(img_embs.device)

            score = (score_i2q - score_i2k) / self.tau
            sig_img_embs = []
            for idx in range(img_embs.size()[0]):
                cur_img_emb = img_embs[idx]
                cur_score = score[idx]
                cur_mask = F.softmax(cur_score, dim=-1).unsqueeze(-1)
                cur_sig_img_emb = cur_mask * cur_img_emb
                cur_sig_img_emb = cur_sig_img_emb.sum(0)
                sig_img_embs.append(cur_sig_img_emb)
                del cur_img_emb, cur_score, cur_mask, cur_sig_img_emb
            img_embs = torch.stack(sig_img_embs, dim=0)
        else:
            score_i2q = F.cosine_similarity(q, iq, dim=-1)
            score_i2k = F.cosine_similarity(k, ik, dim=-1)
        
            score = (score_i2q - score_i2k) / self.tau
            mask = F.softmax(score, dim=-1).unsqueeze(-1)
            img_embs = mask * img_embs
            img_embs = img_embs.sum(1)

        return img_embs

def build_composer_projector(config, composer_type, **kwargs):
    if composer_type == 'single_token':
        modules = [nn.Linear(config.composer_proj_dim, config.hidden_size)]
        mlp_depth = getattr(config, 'composer_proj_mlp_depth', 2)
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    elif 'average_pooling' in str(composer_type):
        return None

    elif 'key_frame_pooling' in str(composer_type):
        if 'merge' in composer_type:
            return KeyFrameProjector()
        return None

    elif 'sig_pooling' in str(composer_type):
        return SigPooling(config)

    elif 'uniform_sample' in str(composer_type):
        return None

    raise ValueError(f"Unknown projector type: {composer_type}")
