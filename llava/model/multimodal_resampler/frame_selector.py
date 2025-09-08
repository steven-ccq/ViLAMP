# import sys
# sys.path.append('/ossfs/workspace/LLaVA-NeXT/llava/model/multimodal_resampler/clip')
# from configuration_clip import CLIPConfig
# from modeling_clip import CLIPModel
# from processing_clip import CLIPProcessor

import torch
import transformers
from transformers import CLIPConfig, CLIPModel, CLIPProcessor
from PIL import Image
from torch import nn

from typing import Union
from sentence_transformers import util
import cv2
import numpy as np
import tqdm

from llava.utils import rank0_print

import math
from sklearn.cluster import KMeans

def build_frame_selector(config):
    frame_selector_type = getattr(config, "frame_selector_type", "SeqFrameSelector")
    if frame_selector_type == "SeqFrameSelector":
        return SeqFrameSelector(config)
    elif frame_selector_type == "ClusterFrameSelector":
        return ClusterFrameSelector(config)
    else:
        raise ValueError(f"Unknown frame selector type: {frame_selector_type}")

class SeqFrameSelector(nn.Module):
    def __init__(self, config):
        super().__init__()

        clip_model_path = getattr(config, "clip_model_path", "clip-ViT-B-32/0_CLIPModel")
        clip_config = CLIPConfig.from_pretrained(clip_model_path)
        self.model = CLIPModel(clip_config)

        self.f2f_thrd = getattr(config, "f2f_thrd", 0.85)
        self.f2t_thrd = getattr(config, "f2t_thrd", -1)
        self.max_frame_num = getattr(config, "max_frame_num", 32)

    def process(self, image_inputs, text_inputs):
        img_embs = self.encode_image(image_inputs)
        text_embs = self.encode_text(text_inputs)
        f2f_scores = self.cos_sim(img_embs, img_embs)
        f2t_scores = self.cos_sim(img_embs, text_embs).reshape(-1)
        return (img_embs, text_embs, f2f_scores, f2t_scores)

    def forward(self,
               image_inputs,
               text_inputs
            ) -> Union[torch.Tensor, torch.Tensor]:
        device = self.model.device
        img_embs, text_embs, f2f_scores, f2t_scores = self.process(image_inputs, text_inputs)
        frame_num = f2t_scores.shape[0]
        is_visited = torch.zeros(frame_num, dtype=bool).to(device)
        is_selected = torch.zeros(frame_num, dtype=int).to(device)

        sorted_scores, sorted_indices = torch.sort(f2t_scores, descending=True)
        idx = 0
        while idx < frame_num and is_selected.sum().item() < self.max_frame_num:
            if sorted_scores[idx] < self.f2t_thrd:
                break
            cur_frame = sorted_indices[idx].item()
            if is_visited[cur_frame]:
                idx += 1
                continue
            is_visited[cur_frame] = True
            is_selected[cur_frame] = 1
            is_similar = f2f_scores[cur_frame] > self.f2f_thrd
            is_visited |= is_similar
            idx += 1

        is_selected = self.distribute_frames(is_selected.tolist(), self.max_frame_num)
        is_selected = torch.tensor(is_selected, dtype=int).to(device)
        return is_selected, f2t_scores, img_embs, text_embs

    def encode_image(self, inputs):
        inputs = inputs.to(self.model.device)
        embs = self.model.get_image_features(**inputs)
        return embs

    def encode_text(self, inputs):
        inputs = inputs.to(self.model.device)
        embs = self.model.get_text_features(**inputs)
        return embs

    def cos_sim(self, img_emb, text_emb):
        cos_scores = util.cos_sim(img_emb, text_emb)
        return cos_scores

    def distribute_frames(self, is_selected, max_frame_num):
        frame_num = len(is_selected)
        existing_frame_num = max_frame_num - sum(is_selected)
        existing = [i for i, val in enumerate(is_selected) if val == 1]

        gaps = []
        if not existing:
            gaps.append((0, frame_num-1))
        else:
            if existing[0] > 0:
                gaps.append((0, existing[0]-1))
            for i in range(1, len(existing)):
                if existing[i-1]+1 <= existing[i]-1:
                    gaps.append((existing[i-1]+1, existing[i]-1))
            if existing[-1] < frame_num-1:
                gaps.append((existing[-1]+1, frame_num-1))

        def total_k(D):
            total =0
            for start, end in gaps:
                length = end - start +1
                if length <= 0:
                    continue
                k = math.ceil(length / D) -1
                total +=k
            return total

        left = 1
        right = frame_num
        while left < right:
            mid = (left + right) // 2
            required = total_k(mid)
            if required <= existing_frame_num:
                right = mid
            else:
                left = mid +1
        D = left

        sum_k = 0
        assigned = []
        for start, end in gaps:
            length = end - start + 1
            if length <= 0:
                continue
            k = math.ceil(length / D) - 1
            if k > 0:
                sum_k += k
                step = math.ceil(length / (k+1))
                positions = []
                for i in range(k):
                    pos = start + step * (i+1)
                    if pos > end:
                        pos = end
                    positions.append(pos)
                assigned.append((start, end, positions))

        s_prime = is_selected.copy()
        for start, end, positions in assigned:
            for pos in positions:
                s_prime[pos] = 1

        remaining = existing_frame_num - sum_k
        if remaining > 0:
            for start, end in reversed(gaps):
                for pos in range(end, start-1, -1):
                    if s_prime[pos] == 0:
                        s_prime[pos]= 1
                        remaining -= 1
                        if remaining == 0:
                            break
                if remaining == 0:
                    break
        return s_prime


class ClusterFrameSelector(nn.Module):
    def __init__(self, config):
        super().__init__()

        clip_model_path = getattr(config, "clip_model_path", "clip-ViT-B-32/0_CLIPModel")
        clip_config = CLIPConfig.from_pretrained(clip_model_path)
        self.model = CLIPModel(clip_config)

        self.cluster_num = getattr(config, "cluster_num", 64)
        self.max_frame_num = getattr(config, "max_frame_num", 32)

    def process(self, image_inputs, text_inputs):
        img_embs = self.encode_image(image_inputs)
        text_embs = self.encode_text(text_inputs)
        f2f_scores = self.cos_sim(img_embs, img_embs)
        f2t_scores = self.cos_sim(img_embs, text_embs).reshape(-1)
        return (img_embs, text_embs, f2f_scores, f2t_scores)

    def forward(self,
               image_inputs,
               text_inputs
            ) -> Union[torch.Tensor, torch.Tensor]:
        device = self.model.device
        img_embs, text_embs, f2f_scores, f2t_scores = self.process(image_inputs, text_inputs)
        frame_num = f2t_scores.shape[0]
        is_selected = torch.zeros(frame_num, dtype=int).to(device)

        cluster_labels = self.cluster(img_embs)
        cluster_tops = [[-100, -1]] * len(cluster_labels)
        for idx, label in enumerate(cluster_labels):
            cur_f2t_score = f2t_scores[idx].item()
            if cur_f2t_score > cluster_tops[label][0]:
                cluster_tops[label] = [cur_f2t_score, idx]
        cluster_tops = [_ for _ in cluster_tops if _[1] != -1]
        cluster_tops = sorted(cluster_tops, key=lambda x: x[0], reverse=True)[:self.max_frame_num]
        selected_idx = [_[1] for _ in cluster_tops]
        is_selected[selected_idx] = 1

        return is_selected, f2t_scores, img_embs

    def cluster(self, img_embs, random_state=0):
        if type(img_embs) == list:
            embeddings_tensor = torch.stack(img_embs)
        embeddings_tensor = img_embs.cpu().float().numpy()
        num_samples = embeddings_tensor.shape[0]
        if self.cluster_num >= num_samples:
            return [cluster_idx for cluster_idx in range(num_samples)]

        kmeans = KMeans(n_clusters=self.cluster_num, random_state=random_state)
        kmeans.fit(embeddings_tensor)
        labels = kmeans.labels_

        return labels

    def encode_image(self, inputs):
        inputs = inputs.to(self.model.device)
        embs = self.model.get_image_features(**inputs)
        return embs

    def encode_text(self, inputs):

        inputs = inputs.to(self.model.device)
        embs = self.model.get_text_features(**inputs)
        return embs

    def cos_sim(self, img_emb, text_emb):
        cos_scores = util.cos_sim(img_emb, text_emb)
        return cos_scores
