import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import logging
import copy
from tqdm import tqdm
import clip

from models.base import BaseLearner
from utils.data_manager import DataManager, FeatureDataset
from torch.utils.data import DataLoader
from utils.grouped_prototypical_net import PrototypicalNet_Grouped
from utils.toolkit import accuracy
from utils.maxsplit import max_cut_split
import random
from collections import Counter
from sklearn.cluster import KMeans
import itertools

class CombinedLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = PrototypicalNet_Grouped(args, pretrained=False, device=self._device)
        self._network = self._network.to(self._device)

        self.data_manager = DataManager(self.args["dataset"], self.args["shuffle"], self.args["seed"], self.args["init_cls"], self.args["increment"])
        self.group_bottlenecks = {}
        self.ori_protos, self.ori_covs = [], []

        self.ptask = 0
        self.mp = {}
        self.cp = {}
        self.mean_dis = {}
        self.p2c = {}
        self.archetypes = {}
        self.gmm_params = {}
        self.group_concept_counts = {}

        self._feature_trainset = None
        self._relations = None
        self.train_loader = None
        self.test_loader = None

    def incremental_train(self):
        self._cur_task += 1
        self._total_classes = self._known_classes + self.data_manager.get_task_size(self._cur_task)
        task_size = self._total_classes - self._known_classes

        logging.info(f"Learning on task {self._cur_task} ({task_size} new classes): {self._known_classes}-{self._total_classes-1}")

        train_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory = True)

        task_test_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        self.task_test_loader = DataLoader(task_test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        self.building_protos()

        self.group_classes(task_size)
        for group_id, classes_in_group in self.p2c.items():
          if not classes_in_group:
              continue

          num_concepts = len(classes_in_group) * self.args.get("pool")

          self.group_concept_counts[group_id] = int(num_concepts)

        updated_group_ids = self._network.update_group_heads_and_explainers(self.p2c, self.group_concept_counts)

        self._build_feature_set(updated_group_ids)
        feature_train_loader = DataLoader(self._feature_trainset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args['num_workers'], pin_memory = True)

        self._train_model(feature_train_loader, self.task_test_loader, updated_group_ids)


    def group_classes(self, task_size):
        logging.info("Part 1: Grouping new classes...")
        self._network.eval()
        new_class_indices = list(range(self._known_classes, self._total_classes))

        grouping_loader = DataLoader(self.train_loader.dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
        raw_features_list, labels_list = [], []
        with torch.no_grad():
            for _, images, targets in grouping_loader:
                raw_features = self._network.extract_vector_raw(images.to(self._device)).float()
                raw_features_list.append(raw_features.cpu())
                labels_list.append(targets.cpu())

        features = torch.cat(raw_features_list)
        labels = torch.cat(labels_list)

        for class_idx in new_class_indices:
            data_index = (labels == class_idx).nonzero().squeeze(-1)
            self.cp[class_idx] = features[data_index].mean(0)

        MAX_CLASSES_FOR_EXACT_CUT = 20

        if task_size > MAX_CLASSES_FOR_EXACT_CUT:
            logging.info(f"Task has {task_size} classes, exceeding threshold of {MAX_CLASSES_FOR_EXACT_CUT}.")
            logging.info("Applying iterative Max-Cut strategy.")
            CHUNK_SIZE = 10

            for i in range(0, task_size, CHUNK_SIZE):
                chunk_indices = new_class_indices[i : i + CHUNK_SIZE]
                current_chunk_size = len(chunk_indices)

                if current_chunk_size == 0:
                    continue

                logging.info(f"Processing chunk of {current_chunk_size} classes: {chunk_indices}")

                if current_chunk_size <= 5 or current_chunk_size % 2 != 0:
                    self.ptask += 1
                    self.p2c[self.ptask] = chunk_indices
                    for class_idx in chunk_indices: self.mp[class_idx] = self.ptask
                    logging.info(f"Chunk is odd-sized. Created one new group {self.ptask} for it.")
                else:
                    class_map = {real_idx: graph_idx for graph_idx, real_idx in enumerate(chunk_indices)}
                    edges, weights = [], []
                    for class_i, class_j in itertools.combinations(chunk_indices, 2):
                        similarity = F.cosine_similarity(self.cp[class_i], self.cp[class_j], dim=0).item()
                        u, v = class_map[class_i], class_map[class_j]
                        edges.append((u, v))
                        weights.append(similarity)

                    _, best_assignment = max_cut_split(current_chunk_size, edges, weights)

                    group_0_classes = [chunk_indices[i] for i, label in enumerate(best_assignment) if label == 0]
                    group_1_classes = [chunk_indices[i] for i, label in enumerate(best_assignment) if label == 1]

                    self.ptask += 1
                    self.p2c[self.ptask] = group_0_classes
                    for class_idx in group_0_classes: self.mp[class_idx] = self.ptask
                    logging.info(f"Split chunk into new group {self.ptask} with classes: {group_0_classes}")

                    self.ptask += 1
                    self.p2c[self.ptask] = group_1_classes
                    for class_idx in group_1_classes: self.mp[class_idx] = self.ptask
                    logging.info(f"And new group {self.ptask} with classes: {group_1_classes}")

        else:
            logging.info(f"Task has {task_size} classes. Using standard grouping strategy.")
            if task_size <= 5 or task_size % 2 != 0:
                logging.info("Task size is odd. Creating one new group.")
                self.ptask += 1
                self.p2c[self.ptask] = new_class_indices
                for class_idx in new_class_indices: self.mp[class_idx] = self.ptask
            else:
                logging.info("Task size is even. Splitting into two groups via Max-Cut.")
                class_map = {real_idx: graph_idx for graph_idx, real_idx in enumerate(new_class_indices)}
                edges, weights = [], []
                for class_i, class_j in itertools.combinations(new_class_indices, 2):
                    similarity = F.cosine_similarity(self.cp[class_i], self.cp[class_j], dim=0).item()
                    u, v = class_map[class_i], class_map[class_j]
                    edges.append((u, v))
                    weights.append(similarity)

                _, best_assignment = max_cut_split(task_size, edges, weights)

                group_0_classes = [new_class_indices[i] for i, label in enumerate(best_assignment) if label == 0]
                group_1_classes = [new_class_indices[i] for i, label in enumerate(best_assignment) if label == 1]

                self.ptask += 1
                self.p2c[self.ptask] = group_0_classes
                for class_idx in group_0_classes: self.mp[class_idx] = self.ptask

                self.ptask += 1
                self.p2c[self.ptask] = group_1_classes
                for class_idx in group_1_classes: self.mp[class_idx] = self.ptask

        logging.info(f"Grouping complete. Current p2c map: {self.p2c}")


    def building_protos(self):
        logging.info("Building prototypes and covariance for new classes...")
        with torch.no_grad():
            loader = DataLoader(self.train_loader.dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

            train_features, train_labels = self.get_image_embeddings(loader)

            train_features = torch.from_numpy(train_features).to(self._device)
            train_labels = torch.from_numpy(train_labels).to(self._device)

            for i in range(self._known_classes, self._total_classes):
                index_mask = (train_labels == i)
                if not index_mask.any():
                    self.ori_protos.append(torch.zeros(self._network.feat_dim, device=self._device))
                    self.ori_covs.append(torch.eye(self._network.feat_dim, device=self._device))
                    continue

                index = torch.nonzero(index_mask).squeeze()
                if index.dim() == 0:
                    index = index.unsqueeze(0)

                class_data = train_features[index]
                cls_mean = class_data.mean(dim=0)
                class_data = train_features[index]
                num_archetypes = self.args.get("num_archetypes", 4)

                kmeans = KMeans(n_clusters=num_archetypes, random_state=self.args['seed'], n_init='auto').fit(class_data.cpu().numpy())
                self.archetypes[i] = torch.from_numpy(kmeans.cluster_centers_).to(self._device)

                cluster_labels = kmeans.labels_

                params_for_class = []
                total_points = len(class_data)
                for cluster_id in range(num_archetypes):
                    points_in_cluster = class_data[cluster_labels == cluster_id]

                    if len(points_in_cluster) < 2:
                        continue

                    weight = len(points_in_cluster) / total_points
                    mean = points_in_cluster.mean(dim=0)
                    cov = torch.cov(points_in_cluster.T)

                    params_for_class.append((weight, mean, cov))

                self.gmm_params[i] = params_for_class
                if class_data.shape[0] > 1:
                    cls_cov = torch.cov(class_data.T) + 1e-4 * torch.eye(class_data.shape[1], device=self._device)
                else:
                    cls_cov = torch.eye(class_data.shape[1], device=self._device)

                self.ori_protos.append(cls_mean)
                self.ori_covs.append(cls_cov)
    def _generate_gaussian_samples_gmm(self, class_idx, num_samples):
      if num_samples == 0 or class_idx not in self.gmm_params:
          return None

      params_for_class = self.gmm_params[class_idx]
      all_samples = []

      for weight, mean, cov in params_for_class:
          num_sub_samples = int(round(num_samples * weight))
          if num_sub_samples == 0:
              continue

          sub_samples = self._sample_from_single_gaussian(mean, cov, num_sub_samples)
          if sub_samples is not None:
              all_samples.append(sub_samples)

      if not all_samples:
          return None

      return torch.cat(all_samples, dim=0)
    def _generate_interpolation_samples(self, class_idx, num_samples):
        if num_samples == 0:
            return None

        if class_idx not in self.archetypes or len(self.archetypes[class_idx]) == 0:
            logging.warning(f"No archetypes found for class {class_idx}. Skipping interpolation.")
            return None

        class_archetypes = self.archetypes[class_idx]
        k = class_archetypes.shape[0]

        weights = torch.rand(num_samples, k, device=self._device)
        weights = F.normalize(weights, p=1, dim=1)

        samples = torch.sum(weights.unsqueeze(2) * class_archetypes.unsqueeze(0), dim=1)
        return samples

    def _sample_from_single_gaussian(self, mean, cov, num_samples):

      if num_samples <= 0:
          return None
      cov = cov + 1e-4 * torch.eye(cov.shape[0], device=self._device)

      try:
          scale_tril = torch.linalg.cholesky(cov)
          rand_norm = torch.randn(num_samples, mean.shape[0], device=self._device)

          samples = mean + rand_norm @ scale_tril.T
          return samples

      except torch.linalg.LinAlgError:
          logging.warning(f"Cholesky decomposition failed for a sub-cluster. Skipping sampling for this component.")
          return None

    def _build_feature_set(self, updated_group_ids):

        logging.info("Building feature set using Balanced Strategy (Interpolation + Gaussian)...")
        vectors_train = []
        labels_train = []

        loader = DataLoader(self.train_loader.dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
        real_vectors, real_labels = self.get_image_embeddings(loader)
        vectors_train.append(real_vectors)
        labels_train.append(real_labels)
        labels_all = np.concatenate(labels_train)
        print(labels_all.shape[0])
        if self._cur_task > 0:
            logging.info("Generating pseudo-features for old classes via mixing...")
            num_classes = self._total_classes - self._known_classes
            num_samples_per_class = self.args.get("samples_per_old_class", int(labels_all.shape[0]/num_classes)) + self.args.get("augment", 25)
            interpolation_ratio = self.args.get("interpolation_ratio", 0.7)

            num_interp_samples = int(num_samples_per_class * interpolation_ratio)
            num_gauss_samples = num_samples_per_class - num_interp_samples


            for class_idx in range(self._known_classes):

                interp_samples = self._generate_interpolation_samples(class_idx, num_interp_samples)

                gauss_samples = self._generate_gaussian_samples_gmm(class_idx, num_gauss_samples)

                if interp_samples is not None and gauss_samples is not None:
                    class_samples = torch.cat([interp_samples, gauss_samples], dim=0)
                elif interp_samples is not None:
                    class_samples = interp_samples
                elif gauss_samples is not None:
                    class_samples = gauss_samples
                else:
                    continue

                vectors_train.append(class_samples.cpu().numpy())
                labels_train.append(np.full(len(class_samples), class_idx))

        vectors_train = np.concatenate(vectors_train)
        labels_train = np.concatenate(labels_train)

        self._feature_trainset = FeatureDataset(vectors_train, labels_train)

    def _train_model(self, feature_train_loader, test_loader, updated_group_ids):
      all_group_ids = list(self._network.group_id_to_head_idx.keys())
      logging.info(f"Starting model training. Training ALL {len(all_group_ids)} groups.")

      if self._cur_task == 0:
          epochs = self.args["FB_epoch"]
          lr = self.args["FB_lr_init"]
      else:
          epochs = self.args["FB_epoch_inc"]
          lr = self.args["FB_lr_inc"]

      self._network.eval()
      for param in self._network.parameters():
          param.requires_grad = False

      trainable_params = []
      logging.info(f"Unfreezing components for ALL groups: {all_group_ids}")
      for group_id in all_group_ids:
          if str(group_id) in self._network.group_explainers:
              explainer = self._network.group_explainers[str(group_id)]
              for param in explainer.parameters():
                  param.requires_grad = True
                  trainable_params.append(param)
          if group_id in self._network.group_id_to_head_idx:
              head_idx = self._network.group_id_to_head_idx[group_id]
              head = self._network.group_heads[head_idx]
              for param in head.parameters():
                  param.requires_grad = True
                  trainable_params.append(param)

      if not trainable_params:
          logging.warning("No trainable parameters found. Skipping training.")
          return

      optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=self.args.get("weight_decay", 0))
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.get("milestones", [epochs//2]), gamma=0.1)
      loss_cls = nn.CrossEntropyLoss()
      best_acc = 0.0
      best_model_state = copy.deepcopy(self._network.state_dict())
      random_class_order_list = list(range(self._known_classes))

      for epoch in range(epochs):
          self._network.train()
          total_loss_val, correct, total = 0, 0, 0

          for idx, (inputs, targets) in enumerate(tqdm(feature_train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
              inputs, targets = inputs.float().to(self._device), targets.long().to(self._device)

              if self._cur_task > 0 and self.args.get('sg_num', 0) > 0:
                  sg_inputs, sg_targets = self._sample_gussian(idx, random_class_order_list, self.args['sg_num'])
                  if sg_inputs.nelement() > 0:
                      inputs = torch.cat([inputs, sg_inputs], dim=0)
                      targets = torch.cat([targets, sg_targets], dim=0)

              optimizer.zero_grad()

              outputs = self._network.forward_cbm(inputs)
              logits = outputs["logits"]

              total_sparsity_loss = 0.0
              group_targets = torch.tensor([self.mp[t.item()] for t in targets], device=self._device)

              active_groups_in_batch = torch.unique(group_targets)

              for group_id_tensor in active_groups_in_batch:
                group_id = group_id_tensor.item()

                head_idx = self._network.group_id_to_head_idx.get(group_id)
                if head_idx is not None:
                    head = self._network.group_heads[head_idx]
                    total_sparsity_loss += torch.linalg.norm(head.weight, ord=1)

              classification_loss = loss_cls(logits, targets)

              loss = classification_loss + \
                    self.args.get("sparse_coeff", 0.001) * total_sparsity_loss

              loss.backward()
              optimizer.step()

              total_loss_val += loss.item()
              _, preds = torch.max(logits, dim=1)
              correct += (preds == targets).sum().item()
              total += len(targets)

          scheduler.step()

          train_acc = correct * 100 / total if total > 0 else 0
          if epoch % self.args["print_freq"] == 0 or epoch == epochs - 1:
              current_acc = self._compute_accuracy_cbm(test_loader)
              logging.info(f"Task: {self._cur_task}, Epoch: {epoch+1}, Train Loss: {total_loss_val/len(feature_train_loader):.6f}, "
                          f"Train Acc: {train_acc:.4f}, Test Acc on New: {current_acc:.4f}")

              if current_acc > best_acc:
                  best_acc = current_acc
                  best_model_state = copy.deepcopy(self._network.state_dict())
                  logging.info(f"New best accuracy found: {best_acc:.2f}%")

      logging.info(f"Loading best model for task {self._cur_task} with Test Acc on New: {best_acc:.2f}%")
      self._network.load_state_dict(best_model_state)

    def eval_task(self, save_conf=False):
        full_test_dataset = self.data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(full_test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if hasattr(self.test_loader.dataset, 'set_mode'):
            self.test_loader.dataset.set_mode('test')

        y_pred, y_true = self._eval_cnn(self.test_loader)

        task_size = self.data_manager.get_task_size(self._cur_task) if self._cur_task > 0 else self._total_classes

        cnn_accy_dict = accuracy(y_pred[:, 0], y_true, self._known_classes, task_size, class_groups=self.p2c)

        if 'total' in cnn_accy_dict:
            cnn_accy_dict['top1'] = cnn_accy_dict['total']

        topk_correct = (y_pred == y_true[:, np.newaxis]).sum()
        topk_acc = np.around((topk_correct / len(y_true)) * 100, decimals=2)

        cnn_accy_dict[f'top{self.topk}'] = topk_acc

        final_accy = cnn_accy_dict.copy()
        final_accy['grouped'] = cnn_accy_dict

        logging.info(f"CNN accuracy: {final_accy}")

        nme_accy = None

        return final_accy, nme_accy


    def after_task(self):
        self._known_classes = self._total_classes

    def _shrink_cov(self,cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag*mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0], device=cov.device)
        alpha1 = 1
        alpha2  = 1
        cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
        return cov_

    def _sample(self,mean, cov, size, shrink=False):
        vec = torch.randn(size, mean.shape[-1], device=self._device)
        if shrink:
            cov = self._shrink_cov(cov)
        sqrt_cov = torch.linalg.cholesky(cov)
        vec = vec @ sqrt_cov.t()
        vec = vec + mean
        return vec

    def _sample_gussian(self,batch_id,random_class_order_list,sg_num):
        sg_inputs = []
        sg_targets = []

        num_old_classes = len(random_class_order_list)
        if num_old_classes == 0:
            return torch.tensor([]).to(self._device), torch.tensor([]).to(self._device)

        class_idx1 = random_class_order_list[batch_id * 2 % num_old_classes]
        class_idx2 = random_class_order_list[(batch_id * 2 + 1) % num_old_classes]
        list_for_one_batch = [class_idx1, class_idx2]

        for i in list_for_one_batch:
            sg_inputs.append(self._sample(self.ori_protos[i], self.ori_covs[i],int(sg_num), shrink=False))
            sg_targets.append(torch.ones(int(sg_num), dtype=torch.long, device=self._device)*i)

        sg_inputs = torch.cat(sg_inputs, dim=0)
        sg_targets = torch.cat(sg_targets, dim=0)

        return sg_inputs, sg_targets
    def get_image_embeddings(self, loader):
        self._network.eval()
        features, labels = [], []
        with torch.no_grad():
            for _, images, targets in tqdm(loader, desc="Extracting features"):
                images = images.to(self._device)
                image_features = self._network.extract_vector(images).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)

                features.append(image_features.cpu())
                labels.append(targets.cpu())

        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()
        return features, labels

    def _compute_accuracy_cbm(self, loader):
        self._network.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for _, inputs, targets in loader:
                features = self._network.extract_vector(inputs.to(self._device)).float()
                features_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
                outputs = self._network.forward_cbm(features_norm)
                logits = outputs["logits"]

                _, preds = torch.max(logits, dim=1)

                correct += (preds.cpu() == targets).sum().item()
                total += len(targets)
        self._network.train()
        return (correct / total) * 100 if total > 0 else 0

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)

                features = self._network.extract_vector(inputs).float()
                features_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

                outputs = self._network.forward_cbm(features_norm)
                logits = outputs["logits"]

                predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]

                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    def _encode_text_attributes(self, attributes):
        self._network.eval()
        attribute_embeddings = []
        with torch.no_grad():
            for i in range((len(attributes) // self.args["batch_size"]) + 1):
                sub_attributes = attributes[i * self.args["batch_size"]: (i + 1) * self.args["batch_size"]]
                if not sub_attributes: continue

                tokens = clip.tokenize([self.data_manager.get_prefix(self.args) + attr for attr in sub_attributes]).to(self._device)

                embeddings = self._network.convnet.encode_text(tokens)
                attribute_embeddings.append(embeddings.cpu())
        return torch.cat(attribute_embeddings).float()
