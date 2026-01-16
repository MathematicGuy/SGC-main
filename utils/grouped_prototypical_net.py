
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import logging
import copy
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import PrototypicalNet, Gateway 
from utils.data_manager import DataManager, FeatureDataset, Pesudo_FeatureDataset
from torch.utils.data import DataLoader

class PrototypicalNet_Grouped(PrototypicalNet):
    def __init__(self, args, pretrained, device):
        super().__init__(args, pretrained, device)
        
        del self.explainer
        self.group_explainers = nn.ModuleDict()

        del self.unity
        self.group_heads = nn.ModuleList()
        
        self.group_to_class_map = {}  
        self.group_id_to_head_idx = {} 
        self.group_class_sorted_cache = {}
    def generate_explainer(self, in_dim, out_dim, bias=True):
      return nn.Linear(in_dim, out_dim, bias=bias)
    def update_group_heads_and_explainers(self, p2c, group_concept_counts):

        new_group_map = copy.deepcopy(p2c)
        
        new_group_explainers = nn.ModuleDict()
        new_group_heads = nn.ModuleList()
        new_group_id_to_head_idx = {}
        updated_group_ids = set()

        old_heads_map = self.group_id_to_head_idx
        old_heads = self.group_heads

        

        for group_id, classes_in_group in new_group_map.items():
            
            num_concepts_in_group = group_concept_counts[group_id]            
            if str(group_id) in self.group_explainers:
                old_explainer = self.group_explainers[str(group_id)]
                old_num_concepts = old_explainer.out_features
                
                new_explainer = self.generate_explainer(self.feat_dim, num_concepts_in_group, bias=True).to(self.device)
                new_explainer.weight.data[:old_num_concepts, :] = old_explainer.weight.data
                
                if old_num_concepts != num_concepts_in_group:
                    updated_group_ids.add(group_id)
            else: 
                new_explainer = self.generate_explainer(self.feat_dim, num_concepts_in_group, bias=False).to(self.device)
                updated_group_ids.add(group_id) 
                
            new_group_explainers[str(group_id)] = new_explainer

            num_classes_in_group = len(classes_in_group)
            new_head = nn.Linear(num_concepts_in_group, num_classes_in_group, bias=False).to(self.device)
            
            if group_id in old_heads_map:
                # logging.info(f"Transferring weights for group {group_id}...")
                old_head_idx = old_heads_map[group_id]
                old_head = old_heads[old_head_idx]
                old_out_features, old_in_features = old_head.weight.shape
                
                copy_out = min(old_out_features, new_head.weight.shape[0])
                copy_in = min(old_in_features, new_head.weight.shape[1])
                new_head.weight.data[:copy_out, :copy_in] = old_head.weight.data[:copy_out, :copy_in]
                
                if old_out_features != num_classes_in_group:
                    # logging.info(f"Group {group_id} was updated with new classes.")
                    updated_group_ids.add(group_id)
            else:
                # logging.info(f"Group {group_id} is a new group.")
                updated_group_ids.add(group_id) 

            new_group_heads.append(new_head)
            new_group_id_to_head_idx[group_id] = len(new_group_heads) - 1

        self.group_explainers = new_group_explainers
        self.group_heads = new_group_heads
        self.group_to_class_map = new_group_map
        self.group_id_to_head_idx = new_group_id_to_head_idx
        
        self.group_class_sorted_cache = {
            gid: sorted(classes)
            for gid, classes in self.group_to_class_map.items()
        }

        return list(updated_group_ids)

    def extract_vector_raw(self, x):
        return self.convnet.encode_image(x)
    def forward_cbm(self, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        total_classes = sum(len(c) for c in self.group_to_class_map.values())

        batch_size = image_features.size(0) 
        final_logits = torch.full((batch_size, total_classes), -np.inf).to(self.device)
        all_concept_scores = {} 
        for group_id, classes_in_group in self.group_to_class_map.items():
            explainer = self.group_explainers[str(group_id)]
            head_idx = self.group_id_to_head_idx[group_id]
            head = self.group_heads[head_idx]

            # 1. Image features -> Group-specific concept scores
            concept_scores = explainer(image_features)
            all_concept_scores[group_id] = concept_scores 

            group_logits = head(concept_scores)
            
            sorted_classes = self.group_class_sorted_cache[group_id]
            final_logits[:, sorted_classes] = group_logits

        return {"logits": final_logits, "concept_scores_dict": all_concept_scores}





