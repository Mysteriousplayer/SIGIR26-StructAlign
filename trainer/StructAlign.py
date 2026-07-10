import os
from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict, deque
import time

from trainer.base_trainer import BaseTrainer
from modules.optimization import AdamW
from datasets.utils.cached_video_dataset import ReferenceVideoDataset, RefDataIterator
from modules.trainer_utils import log_training_progress, load_stored_embed, update_exp_result
from evaluator.validator import Validator
import math
import torch.nn.functional as F
import torch.nn as nn
from modules.loss import compute_etf_alignment_loss_base, compute_etf_alignment_loss_incremental

class Trainer(BaseTrainer):
    def __init__(self, ref_model, model, loss, metrics, current_task_id, config, 
                 train_data_loader, valid_data_loader, tokenizer, list_val_acc_ii, 
                 num_epochs, experiment=None,state=None):
        """
        Initialize the Trainer and configure model-specific settings.
        """
        # Call the parent constructor (BaseTrainer handles generic setup)
        super().__init__(model, loss, metrics, current_task_id, num_epochs, config)
        
        self.current_task_id = current_task_id
        self.config = config
        self.num_epochs = num_epochs
        self.experiment = experiment
        self.tokenizer = tokenizer
        self.list_val_acc_ii = list_val_acc_ii
        self.best = -1.0
        self.task_best = -1.0
        self.step = 0
        self.total_epochs = num_epochs + 1
        self.start_epoch = 1
        self.num_tasks = config.task_num

        # Initialize Accelerator and set gradient accumulation steps
        self._initialize_accelerator()
        
        # Prepare the reference model (set to eval mode)
        self._prepare_reference_model(ref_model)
        
        # Assign data loaders
        self.train_data_loader = train_data_loader
        self.val_loaders_list = valid_data_loader
        
        # Freeze model parameters (model-specific freezing strategy)
        self._freeze_model_parameters()
        
        # Configure optimizer by grouping parameters and setting learning rates
        self._configure_optimizer()
        
        # Prepare the model, optimizer, and training data loader with Accelerator
        self._prepare_with_accelerator()
        
        # Set up logging files (overall and per-task)
        self._setup_logging()
        
        # Reset model-specific counters (e.g., LoRA counters)
        self._reset_model_counters()
        
        # If using 'triplet' loss and current task > 1, load previous video embeddings
        if self.config.loss == 'triplet' and self.current_task_id > 1:
            self._setup_ref_vid_loader()

        # Keep the original SCL_and_CRP path available while allowing
        # SCL_and_CRP_v2 to reuse the same trainer flow.
        if self.config.loss in ('SCL_and_CRP', 'SCL_and_CRP_v2') and self.current_task_id > 1:
            self._setup_ref_vid_loader()

        # Construct the Validator object for validation *****
        self.validator = Validator(
            self.model, metrics, config, self.current_task_id, valid_data_loader, tokenizer, 
            self.accelerator, experiment, self.checkpoint_dir, self.list_val_acc_ii,
            self.task_log, self.overall_log
        )

        # ETF and prototype bookkeeping are only enabled for the
        # StructAlign branch.
        self.use_structalign_proto = getattr(self.config, 'arch', None) == 'StructAlign'
        self.use_structalign_etf = self.use_structalign_proto and self.config.loss in ('SCL_and_CRP', 'SCL_and_CRP_v2')

        self.prototype_state = state
        self.class_label = []
        self.text_radius = []
        self.text_prototype = []
        self.text_cov_list = []
        self.video_radius = []
        self.video_prototype = []
        self.video_cov_list = []

        self.class_label_c = None
        self.text_radius_c = []
        self.text_prototype_c = None
        self.text_cov_list_c = None
        self.video_radius_c = []
        self.video_prototype_c = None
        self.video_cov_list_c = None

        self.etf = None
        if self.use_structalign_proto:
            self.class_label = self.prototype_state["class_label"]
            self.text_radius = self.prototype_state["text_radius"]
            self.text_prototype = self.prototype_state["text_prototype"]
            self.text_cov_list = self.prototype_state["text_cov_list"]
            self.video_radius = self.prototype_state["video_radius"]
            self.video_prototype = self.prototype_state["video_prototype"]
            self.video_cov_list = self.prototype_state["video_cov_list"]

        if self.use_structalign_etf:
            if self.config.dataset_name == 'ACTNET':
                self.etf = self.get_etf(self.config.embed_dim, 200, device=self.accelerator.device)
            else:
                self.etf = self.get_etf(self.config.embed_dim, 20, device=self.accelerator.device)
            self.etf.requires_grad_(False)


        self.config.enable_probe = False
        self.config.probe_steps_per_epoch = 3     
        self.config.probe_max_batches = 10

        # Debug option: optional sparse diagnostics, disabled by default.
        self.debug_v2 = bool(getattr(self.config, 'debug_v2', False))
        self.debug_v2_interval = max(1, int(getattr(self.config, 'debug_v2_interval', 20)))
        self.debug_v2_verbose = bool(getattr(self.config, 'debug_v2_verbose', False))
        if hasattr(self.loss, 'debug_enabled'):
            self.loss.debug_enabled = False

        # Implementation note: early stop only affects the epoch loop, not loss/checkpoint logic.
        self.early_stop_enabled = bool(getattr(self.config, 'early_stop', False))
        self.early_stop_patience = max(1, int(getattr(self.config, 'early_stop_patience', 5)))
        self.early_stop_warmup_epochs = max(0, int(getattr(self.config, 'early_stop_warmup_epochs', 10)))
        self.early_stop_min_delta = float(getattr(self.config, 'early_stop_min_delta', 0.0))
        self.early_stop_bad_epochs = 0
        self.early_stop_best = -float('inf')

    def _v2_should_debug(self, batch_idx):
        return (
            self.debug_v2
            and getattr(self.config, 'loss', None) == 'SCL_and_CRP_v2'
            and ((batch_idx + 1) % self.debug_v2_interval == 0 or batch_idx == 0)
        )

    def _v2_stat_line(self, prefix, stats):
        if not stats:
            return f"{prefix}=None"
        keys = ['mean', 'std', 'min', 'max', 'diag_mean', 'offdiag_mean', 'diag_minus_offdiag']
        parts = []
        if 'shape' in stats:
            parts.append(f"shape={stats['shape']}")
        for key in keys:
            if key in stats and stats[key] is not None:
                parts.append(f"{key}={stats[key]:.6f}")
        return f"{prefix}: " + ', '.join(parts)

    def _v2_tensor_norm_line(self, name, value):
        if value is None:
            return f"{name}_norm=None"
        with torch.no_grad():
            if isinstance(value, list):
                tensors = [v.detach().float().norm(dim=-1).mean() for v in value if torch.is_tensor(v) and v.numel() > 0]
                if len(tensors) == 0:
                    return f"{name}_norm=None"
                norms = torch.stack(tensors)
            elif torch.is_tensor(value):
                norms = value.detach().float().norm(dim=-1).reshape(-1)
            else:
                return f"{name}_norm=None"
            return f"{name}_norm_mean={norms.mean().item():.6f}, {name}_norm_std={norms.std(unbiased=False).item():.6f}"

    def _v2_print_debug_stats(self, epoch, batch_idx, base_loss, etf_loss, lambda_etf, proto_aug_loss, total_loss, text_embeds, video_embeds, categories):
        if not self._v2_should_debug(batch_idx):
            return
        if hasattr(self.accelerator, 'is_main_process') and not self.accelerator.is_main_process:
            return
        loss_stats = getattr(self.loss, 'last_debug_stats', {}) if hasattr(self.loss, 'last_debug_stats') else {}
        proto_value = proto_aug_loss.detach().float().item() if torch.is_tensor(proto_aug_loss) else None
        unique_categories = torch.unique(categories.detach()).cpu().tolist() if torch.is_tensor(categories) else []
        print(
            f"\n[V2-LOSS] task={self.task_id} epoch={epoch} batch={batch_idx + 1} "
            f"loss_t2v={loss_stats.get('loss_t2v')} loss_crp={loss_stats.get('loss_crp')} "
            f"base_loss={base_loss.detach().float().item():.6f} "
            f"etf_loss={etf_loss.detach().float().item():.6f} "
            f"lambda_etf={lambda_etf.detach().float().item():.6f} "
            f"proto_aug_loss={proto_value} total_loss={total_loss.detach().float().item():.6f} "
            f"categories={unique_categories}"
        )
        print(f"[V2-SIM] {self._v2_stat_line('t2v', loss_stats.get('t2v'))}")
        print(f"[V2-SIM] {self._v2_stat_line('v2t', loss_stats.get('v2t'))}")
        print(f"[V2-WF] {self._v2_stat_line('valid_words', loss_stats.get('wf_valid_words'))}")
        print(f"[V2-WF] {self._v2_stat_line('valid_frames', loss_stats.get('wf_valid_frames'))}")
        print(f"[V2-WF] {self._v2_stat_line('word_to_frame', loss_stats.get('wf_word_to_frame'))}")
        print(f"[V2-WF] {self._v2_stat_line('frame_to_word', loss_stats.get('wf_frame_to_word'))}")
        print(f"[V2-NORM] {self._v2_tensor_norm_line('text', text_embeds)}; {self._v2_tensor_norm_line('video', video_embeds)}")

    def etf_alignment_loss_base(self, text_embeds_list, video_embeds, categories):
        """
        Thin wrapper around modules.loss ETF computation to keep trainer-side
        parameter/state ownership unchanged for experiment stability.
        """
        return compute_etf_alignment_loss_base(
            text_embeds_list=text_embeds_list,
            video_embeds=video_embeds,
            categories=categories,
            etf=self.etf,
            etf_proj1=self.model.etf_proj1,
            etf_proj2=self.model.etf_proj2,
        )

    def etf_alignment_loss_incremental(self, text_embeds_list, video_embeds, categories):
        """
        Thin wrapper around modules.loss ETF computation to keep trainer-side
        parameter/state ownership unchanged for experiment stability.
        """
        text_proto_aug = []
        video_proto_aug = []
        proto_aug_label = []

        video_proto_aug, text_proto_aug, proto_aug_label = self.apa_(
            self.text_prototype, text_proto_aug, self.video_prototype, video_proto_aug, proto_aug_label
        )

        video_proto_aug = torch.from_numpy(np.float32(np.asarray(video_proto_aug))).float().to(self.accelerator.device)
        text_proto_aug = torch.from_numpy(np.float32(np.asarray(text_proto_aug))).float().to(self.accelerator.device)
        proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.accelerator.device)

        return compute_etf_alignment_loss_incremental(
            text_embeds_list=text_embeds_list,
            video_embeds=video_embeds,
            categories=categories,
            etf=self.etf,
            etf_proj1=self.model.etf_proj1,
            etf_proj2=self.model.etf_proj2,
            text_proto_aug=text_proto_aug,
            video_proto_aug=video_proto_aug,
            proto_aug_label=proto_aug_label,
        )

    def apa_(self, text_proto, text_proto_aug, video_proto, video_proto_aug, proto_aug_label):
            video_proto_num = video_proto.shape[0]
            index = list(range(video_proto_num))

            for j in range(self.config.batch_size * 2):
                np.random.shuffle(index)
                video_random_vec = np.random.normal(0, 1, self.config.embed_dim) * self.video_radius[index[0]] * 1
                video_p_feature = video_proto[index[0]] + video_random_vec
                video_proto_aug.append(video_p_feature)

                text_random_vec = np.random.normal(0, 1, self.config.embed_dim) * self.text_radius[index[0]] * 1
                text_p_feature = text_proto[index[0]] + text_random_vec
                text_proto_aug.append(text_p_feature)

                proto_aug_label.append(self.class_label[index[0]])

            return video_proto_aug, text_proto_aug, proto_aug_label
    
    def get_etf(self, feature_dim: int, classes: int, device: torch.device):
        """
        Generate an ETF (Equiangular Tight Frame) basis vector matrix with shape (feature_dim, classes)
        """
        with torch.no_grad():
            g = torch.Generator(device=device)
            g.manual_seed(0)

            # Step1: simplex ETF (C, C)
            I = torch.eye(classes, device=device)
            S = I - torch.ones(classes, classes, device=device) / classes
            S = S * math.sqrt(classes / (classes - 1))  # scale

            # Step2: random projection to feature_dim space
            W = torch.randn(feature_dim, classes, generator=g, device=device)

            # Step3: projected ETF
            etf = W @ S  # (D, C)

            # Step4: L2-normalize each ETF vector
            etf = torch.nn.functional.normalize(etf, dim=0)
        return etf  

    def _freeze_model_parameters(self):
        """
        Apply the freezing strategy for model parameters based on their names 
        and the current task.
        """
        for name, param in self.model.named_parameters():
            if "vision_model" in name:
                param.requires_grad = False
                if "frame_cross_attention" in name or "alpha" in name:
                    param.requires_grad = True
            elif "text_model" in name:
                param.requires_grad = False
                if self.current_task_id == 1:
                    if "lora_A" in name or "lora_Bs.0" in name or "w_noise" in name or "task_prototype" in name:
                        param.requires_grad = True
                else:
                    if "lora" in name or "w_noise" in name or "task_prototype" in name:
                        param.requires_grad = True

    def _configure_optimizer(self):
        """
        Group model parameters and configure the optimizer with different learning
        rates based on parameter type and current task.
        """
        params_optimizer = list(self.model.named_parameters())
        
        self.clip_text_params = [
            p for n, p in params_optimizer 
            if "text_model" in n and ("lora" in n or "w_noise" in n) and p.requires_grad
        ]
        self.clip_vision_params = [
            p for n, p in params_optimizer 
            if "frame_cross_attention" in n and p.requires_grad
        ]
        self.noclip_params = [
            p for n, p in params_optimizer 
            if (("alpha" in n or "task_prototype" in n) and p.requires_grad)
        ]
        
        # Set learning rates based on current task and dataset
        if self.current_task_id == 1:

            if self.config.dataset_name == 'MSRVTT':
                clip_t_lr = 2e-5  # 2e-5   3e-6   
                clip_v_lr = 2e-5  # 2e-5   3e-6   
                noclip_lr = 2e-5  # 2e-5   1e-5   
            elif self.config.dataset_name == 'ACTNET':
                clip_t_lr = 2e-5  # 5e-5    
                clip_v_lr = 2e-5  # 5e-5    
                noclip_lr = 2e-5  # 1e-4    
            else:
                clip_t_lr = 2e-5
                clip_v_lr = 2e-5
                noclip_lr = 2e-5
        else:
            clip_t_lr = float(self.config.clip_t_lr)
            clip_v_lr = float(self.config.clip_v_lr)
            noclip_lr = float(self.config.noclip_lr)


        print(f"clip_text_LR: {clip_t_lr}, clip_vision_LR: {clip_v_lr}, noclip_LR: {noclip_lr}")
        
        optimizer_grouped_params = [
            {'params': self.clip_text_params, 'lr': clip_t_lr, 'name': 'clip_text'},
            {'params': self.clip_vision_params, 'lr': clip_v_lr, 'name': 'clip_vision'},
            {'params': self.noclip_params, 'lr': noclip_lr, 'name': 'non_clip'}
        ]
        
        total_params_size = sum(p.numel() * p.element_size() for p in self.model.parameters() if p.requires_grad)
        print('The number of Total Trainable Parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")
        
        self.optimizer = AdamW(optimizer_grouped_params, weight_decay=self.config.weight_decay)
        # self.optimizer = torch.optim.SGD(optimizer_grouped_params, momentum=0.9, weight_decay=self.config.weight_decay * 0.1)

    def _reset_model_counters(self):
        """
        Reset all LoRA counters in the model (if applicable).
        """
        self.model.reset_all_lora_counters()

    def _setup_ref_vid_loader(self):
        self.ref_vid_embeds = []
        for n_task in range(self.current_task_id - 1):
            task_vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
            self.ref_vid_embeds.append(task_vid_embeds.to(self.accelerator.device))
        
        cached_dataset = ReferenceVideoDataset(self.ref_vid_embeds)
        ref_vid_loader = DataLoader(
            cached_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        self.ref_vid_loader = RefDataIterator(ref_vid_loader, self.device)

    def set_scheduler(self, scheduler):
        self.lr_scheduler = self.accelerator.prepare(scheduler)

    def get_list_val_acc_ii(self):
        return self.list_val_acc_ii
    
    def probe_epoch_features(
        self,
        model,
        train_loader,
        device,
        max_batches=None,
    ):
        was_training = model.training
        model.eval()

        all_text, all_video, all_labels = [], [], []
        val_loader, num_classes = self.val_loaders_list[self.task_id - 1]

        for batch_idx, data in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if self.tokenizer is not None:
                data['text'] = self.tokenizer(
                    data['text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                data['text'] = {k: v.to(device) for k, v in data['text'].items()}

            data['video'] = data['video'].to(device)

            if self.config.dataset_name == 'ACTNET':
                data['category'] = data['category'] + int((self.task_id - 1) * (200 / self.num_tasks))

            
            categories = data['category'].long().to(device)

            text_embeds_list, video_embeds = model(data, image=False)

            prototypes = self.etf[:, categories].T  # (B, D)

            text_reprs, video_reprs = [], []

            for b in range(len(text_embeds_list)):
                w = text_embeds_list[b]
                f = video_embeds[b]
                p = prototypes[b]

                w_proj = model.etf_proj1(w)
                f_proj = model.etf_proj2(f)

                alpha = F.softmax(w_proj @ p, dim=0)
                beta  = F.softmax(f_proj @ p, dim=0)

                text_reprs.append((alpha.unsqueeze(-1) * w_proj).sum(dim=0))
                video_reprs.append((beta.unsqueeze(-1) * f_proj).sum(dim=0))

            text_norm  = F.normalize(torch.stack(text_reprs), dim=-1)
            video_norm = F.normalize(torch.stack(video_reprs), dim=-1)

            all_text.append(text_norm.cpu())
            all_video.append(video_norm.cpu())
            all_labels.append(categories.cpu())
        if was_training:
            model.train()
        return {
            "text_norm":  torch.cat(all_text,  dim=0),
            "video_norm": torch.cat(all_video, dim=0),
            "labels":     torch.cat(all_labels, dim=0),
        }
    def _save_probe(self, epoch, step, probe_data):
        save_dir = os.path.join(self.config.output_dir, "probe")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir, f"task{self.task_id}_epoch{epoch}_step{step}.pt"
        )
 
        torch.save({
            "epoch": epoch,
            "step": step,
            "task_id": self.task_id,
            **probe_data
        }, save_path)


    def train(self, memory_callback=None):
        """
        Full training logic with support for frozen modules
        """
        for epoch in range(self.start_epoch, self.total_epochs):
            epoch_log = self._train_epoch(epoch)
            if isinstance(epoch_log, dict) and epoch_log.get('early_stop', False):
                print(f"Early stopping task {self.task_id} at epoch {epoch}: R@1 did not improve for {self.early_stop_patience} epochs.")
                break
            #################################
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()
            #################################
        # Handle frozen modules if configured
        if hasattr(self.config, 'frozen') and self.config.frozen:
            self._handle_frozen_modules()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
       
        if epoch == 1:
            if self.config.init_validation:
                # *********************
                if self.config.wf:
                    # Inference fusion switch:
                    # false -> original word-frame evaluation path
                    # true  -> fused word-frame + global similarity evaluation path
                    if bool(getattr(self.config, 'use_fusion_v3', False)):
                        self.validator.task_validation_v3(self.task_id, 0)
                    else:
                        self.validator.task_validation_v2(self.task_id, 0)
                else:
                    self.validator.task_validation_ori(self.task_id, 0)
            print("Starting Training...")

        self.model.train()
        total_loss = 0.0
        early_stop_triggered = False
        num_steps = len(self.train_data_loader)

        epoch_start_time = time.time()
        with self.experiment.train():
            for batch_idx, data in enumerate(self.train_data_loader):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.accelerator.device)
                else:
                    data['text'] = {key: val.to(self.accelerator.device) for key, val in data['text'].items()}
                
                data['video'] = data['video'].to(self.accelerator.device)
                if self.config.task_prototype:
                    data['prototype_id'] = self.task_id

                with self.accelerator.autocast():
                    text_embeds, video_embeds = self.model(data, image=False)
                    if self.config.loss == 'NCELearnableTempLoss':
                        loss = self.loss(video_embeds, text_embeds, self.model.clipmodel.logit_scale)
                    elif self.config.loss == 'lwf':
                        with torch.no_grad():
                            ref_text_embeds, ref_video_embeds = self.ref_model(data)
                        loss = self.loss(video_embeds, text_embeds, self.model.clipmodel.logit_scale, ref_video_embeds, ref_text_embeds)
                    elif self.config.loss == 'triplet':
                        if self.task_id > 1:
                            ref_vid_embeds = self.ref_vid_loader.get_next()
                        else:
                            ref_vid_embeds = None    
                        loss = self.loss(video_embeds, text_embeds, self.model.clipmodel.logit_scale, ref_vid_embeds, self.config.loss_scale)
                    # Implementation note: SCL_and_CRP_v2 only swaps the loss implementation; trainer-side flow remains the same.
                    elif self.config.loss in ('SCL_and_CRP', 'SCL_and_CRP_v2'):
                        # Debug option: diagnostics are opt-in and do not change loss flow.
                        debug_this_batch = self._v2_should_debug(batch_idx)
                        if hasattr(self.loss, 'debug_enabled'):
                            self.loss.debug_enabled = debug_this_batch
                        proto_aug_loss = None
                        ########################################################
                        if self.task_id > 1:
                            self.ref_model.eval()
                            for p in self.ref_model.parameters():
                                p.requires_grad = False
                            ref_vid_embeds = self.ref_vid_loader.get_next()
                            with torch.no_grad():
                                ref_text_embeds, ref_video_embeds = self.ref_model(data)
                            proto_aug_loss = self._compute_loss(video_embeds, text_embeds)
                            
                        else:
                            ref_vid_embeds = None    
                            ref_text_embeds = None
                            ref_video_embeds = None

                        loss = self.loss(video_embeds, text_embeds, self.model.clipmodel.logit_scale,self.ref_model.clipmodel.logit_scale, ref_vid_embeds,  ref_video_embeds, ref_text_embeds, self.config.loss_scale)
                        base_loss_for_debug = loss

                        if self.config.dataset_name == 'ACTNET':
                            data['category'] = data['category'] + int((self.task_id-1) * (200/self.num_tasks))
                            
                        categories = data['category'].to(self.accelerator.device)
                        
                        if self.task_id > 1:
                            etf_loss = self.etf_alignment_loss_incremental(text_embeds, video_embeds, categories)
                        else:
                            etf_loss = self.etf_alignment_loss_base(text_embeds, video_embeds, categories)
                        

                        alpha = 0.1  # 
                        lambda_etf = (loss.detach() / etf_loss.detach()) * alpha
                        
                        if self.task_id > 1:
                            loss = loss + lambda_etf * etf_loss + proto_aug_loss * 0.5
                        else:
                            loss = loss + lambda_etf * etf_loss
                        self._v2_print_debug_stats(
                            epoch, batch_idx, base_loss_for_debug, etf_loss,
                            lambda_etf, proto_aug_loss, loss, text_embeds,
                            video_embeds, categories
                        )
                        #############################################################
                    else:
                        raise NotImplementedError(f"Loss {self.config.loss} not implemented")

                    scaled_loss = loss / self.gradient_accumulation_steps

                self.accelerator.backward(scaled_loss)
                # self.model.get_gradient_stats()

                # Log the current loss and learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                total_loss += loss.detach().item() 

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(self.train_data_loader) - 1:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    #####################################
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    #####################################
                    self.optimizer.zero_grad()

                    torch.clamp_(self.model.clipmodel.logit_scale.data, 0, np.log(200))
                    self.step += 1
                    ####################################
                    if self.config.enable_probe:
                        probe_interval = max(
                            1,
                            num_steps // self.config.probe_steps_per_epoch
                        )

                        if batch_idx % probe_interval == 0:
                            # with torch.no_grad():
                            #     probe_data = self.probe_epoch_features(
                            #         model=self.model,
                            #         train_loader=self.train_data_loader,
                            #         device=self.accelerator.device,
                            #         max_batches=self.config.probe_max_batches,
                            #     )
                            with torch.no_grad():
                                probe_data = self.probe_epoch_features(
                                    model=self.model,
                                    train_loader=self.train_data_loader,
                                    device=self.accelerator.device,
                                    max_batches=None,
                                )
                            self._save_probe(epoch, step=self.step, probe_data=probe_data)
                    ####################################
                    # Print progress information
                    log_training_progress(self.experiment, self.task_id, self.step, epoch, batch_idx, num_steps, 
                                        epoch_start_time, loss.detach().item(), current_lr)
            
        # Evaluate on the validation set 
        
        if epoch % self.evals_per_epoch == 0 or epoch == self.total_epochs:
            # *******************
            if self.config.wf:
                # Inference fusion switch:
                    # false -> original word-frame evaluation path
                    # true  -> fused word-frame + global similarity evaluation path
                if bool(getattr(self.config, 'use_fusion_v3', False)):
                    task_res = self.validator.task_validation_v3(self.task_id, epoch)
                else:
                    task_res = self.validator.task_validation_v2(self.task_id, epoch)
            else:
                task_res = self.validator.task_validation_ori(self.task_id, epoch)
            
            if self.config.load_best:
                # First get the overall validation result
                if self.task_id > 1:
                    #****************************
                    if self.config.wf:
                        # Inference fusion switch:
                        # false -> original word-frame evaluation path
                        # true  -> fused word-frame + global similarity evaluation path
                        if bool(getattr(self.config, 'use_fusion_v3', False)):
                            overall_res = self.validator.validate_v3(self.task_id, epoch, self.step)
                        else:
                            overall_res = self.validator.validate_v2(self.task_id, epoch, self.step)
                    else:
                        overall_res = self.validator.validate_ori(self.task_id, epoch, self.step)
                    current_score = overall_res['R1']
                else:
                    # For first task, overall performance equals task performance
                    overall_res = task_res
                    current_score = task_res['R1']
                # Compare using overall score instead of task score
                if current_score >= self.best:
                    self.best = current_score  # Update best with overall score
                    self._save_checkpoint(epoch, save_best=True)
                    self.global_best = overall_res['R1']
                    
                    ####################
                    if self.use_structalign_proto:
                        self.compute_proto()
                    ####################
                    # Log results
                    update_exp_result(self.task_log, self.task_id, 
                                    r1=task_res['R1'], r5=task_res['R5'], 
                                    r10=task_res['R10'], medr=task_res['MedR'], 
                                    meanr=task_res['MeanR'])
                    update_exp_result(self.overall_log, self.task_id, 
                                    r1=overall_res['R1'], r5=overall_res['R5'], 
                                    r10=overall_res['R10'], medr=overall_res['MedR'], 
                                    meanr=overall_res['MeanR'])
                    self.experiment.log_metric("CIL_Performance(R@1)", overall_res['R1'], step=self.task_id)
                    self.experiment.log_metric("CIL_Performance(R@5)", overall_res['R5'], step=self.task_id)

                    print(f"\nCurrent Best Overall R@1 is {self.best:.6f}")
                else:
                    print(f"\nCurrent Overall R@1 is {current_score:.6f} < {self.best:.6f}")
            else:
                self.best = task_res['R1']
                current_score = task_res['R1']
                print(f"\nCurrent Final R@1 is {self.best:.6f}")

        if self.early_stop_enabled and (epoch % self.evals_per_epoch == 0 or epoch == self.total_epochs):
            if current_score > self.early_stop_best + self.early_stop_min_delta:
                self.early_stop_best = current_score
                self.early_stop_bad_epochs = 0
            elif epoch > self.early_stop_warmup_epochs:
                self.early_stop_bad_epochs += 1

            if epoch <= self.early_stop_warmup_epochs:
                print(
                    f"EarlyStop warmup epoch {epoch}/{self.early_stop_warmup_epochs}: "
                    f"R@1={current_score:.6f}, best={self.early_stop_best:.6f}"
                )
            else:
                print(
                    f"EarlyStop monitor R@1={current_score:.6f}, "
                    f"best={self.early_stop_best:.6f}, "
                    f"bad_epochs={self.early_stop_bad_epochs}/{self.early_stop_patience}"
                )
                early_stop_triggered = self.early_stop_bad_epochs >= self.early_stop_patience

        if self.use_structalign_proto and (epoch == (self.total_epochs-1) or early_stop_triggered):
            if self.current_task_id == 1:
                self.prototype_state['text_radius'] = self.text_radius_c
                self.prototype_state['text_prototype'] = np.asarray(self.text_prototype_c)
                self.prototype_state['class_label'] = self.class_label_c
                self.prototype_state['text_cov_list'] = self.text_cov_list_c

                self.prototype_state['video_radius'] = self.video_radius_c
                self.prototype_state['video_prototype'] = np.asarray(self.video_prototype_c)
                self.prototype_state['video_cov_list'] = self.video_cov_list_c
            else:
                self.prototype_state['class_label'] = np.concatenate((self.prototype_state['class_label'], self.class_label_c), axis=0)
                self.prototype_state['text_prototype'] = np.concatenate((self.prototype_state['text_prototype'], self.text_prototype_c), axis=0)
                self.prototype_state['text_radius'] = np.concatenate((self.prototype_state['text_radius'], self.text_radius_c), axis=0)
                self.prototype_state['text_cov_list'] = np.concatenate((self.prototype_state['text_cov_list'], self.text_cov_list_c), axis=0)

                self.prototype_state['video_prototype'] = np.concatenate((self.prototype_state['video_prototype'], np.asarray(self.video_prototype_c)), axis=0)
                self.prototype_state['video_radius'] = np.concatenate((self.prototype_state['video_radius'], self.video_radius_c), axis=0)
                self.prototype_state['video_cov_list'] = np.concatenate((self.prototype_state['video_cov_list'], self.video_cov_list_c), axis=0)
        res = {
            'loss_train':  total_loss / num_steps,
            'early_stop': early_stop_triggered
        }
            
        torch.cuda.empty_cache()
        return res
        
    def _compute_loss(self, video_embeds, text_embeds):
        
        text_proto_aug = []
        video_proto_aug = []
        proto_aug_label = []

        video_proto_aug, text_proto_aug, proto_aug_label = self.apa(self.text_prototype, text_proto_aug, self.video_prototype, video_proto_aug, proto_aug_label)
        video_proto_aug = torch.from_numpy(np.float32(np.asarray(video_proto_aug))).float().to(self.accelerator.device)
        text_proto_aug = torch.from_numpy(np.float32(np.asarray(text_proto_aug))).float().to(self.accelerator.device)

        text_embeds = torch.stack([emb.mean(dim=0) for emb in text_embeds], dim=0)
        video_embeds = video_embeds.mean(dim=1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        video_embeds = F.normalize(video_embeds, dim=-1)

        vis_feat = torch.cat([video_embeds, video_proto_aug],dim=0)
        text_feat = torch.cat([text_embeds, text_proto_aug],dim=0)

        logit_scale = self.model.clipmodel.logit_scale.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss_protoAug = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        

        return loss_protoAug

    def apa(self, text_proto, text_proto_aug, video_proto, video_proto_aug, proto_aug_label):
        video_proto_num = video_proto.shape[0]
        index = list(range(video_proto_num))

        for j in range(self.config.batch_size * 1):
            np.random.shuffle(index)
            video_random_vec = np.random.normal(0, 1, self.config.embed_dim) * self.video_radius[index[0]] * 1
            video_p_feature = video_proto[index[0]] + video_random_vec
            video_proto_aug.append(video_p_feature)

            text_random_vec = np.random.normal(0, 1, self.config.embed_dim) * self.text_radius[index[0]] * 1
            text_p_feature = text_proto[index[0]] + text_random_vec
            text_proto_aug.append(text_p_feature)

            proto_aug_label.append(self.class_label[index[0]])

        return video_proto_aug, text_proto_aug, proto_aug_label
    
    

    def compute_proto(self):
        """
        Overwrite-mode: compute per-category mean/std and save a single snapshot.
        Each call writes a fresh JSON (overwrites if exists), so no duplicate category entries.
        """
        # load checkpoint if provided
        text_features = []
        video_features = []
        labels = []
        self.model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_data_loader):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.accelerator.device)
                else:
                    data['text'] = {key: val.to(self.accelerator.device) for key, val in data['text'].items()}
            
                data['video'] = data['video'].to(self.accelerator.device)
                categories = data['category']
                
                with self.accelerator.autocast():
                    text_embeds, video_embeds = self.model(data, image=False)
                    text_pooled = torch.stack([emb.mean(dim=0) for emb in text_embeds], dim=0)  # (B, D)
                    video_pooled = video_embeds.mean(dim=1)  # (B, D)
                    text_norm = F.normalize(text_pooled, dim=-1)
                    video_norm = F.normalize(video_pooled, dim=-1)

                if video_norm.shape[0] == self.config.batch_size:
                        labels.append(categories.cpu().numpy())
                        text_features.append(text_norm.cpu().numpy())
                        video_features.append(video_norm.cpu().numpy())

        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        
        text_features = np.array(text_features)
        text_features = np.reshape(text_features, (text_features.shape[0] * text_features.shape[1], text_features.shape[2]))
        video_features = np.array(video_features)
        video_features = np.reshape(video_features, (video_features.shape[0] * video_features.shape[1], video_features.shape[2]))
        text_prototype, text_radius,  text_cov_list,  video_prototype, video_radius, video_cov_list, class_label = self.cmp(text_features,video_features, labels, labels_set)

        self.class_label_c = class_label
        self.text_radius_c = text_radius
        self.text_prototype_c = np.asarray(text_prototype)
        self.text_cov_list_c = text_cov_list

        self.video_radius_c = video_radius
        self.video_prototype_c = np.asarray(video_prototype)
        self.video_cov_list_c = video_cov_list

        print('radius:', self.text_radius_c)
        print('v_radius:', self.video_radius_c)
        print('class:', self.class_label_c)
        print('length of conv list:', len(self.text_cov_list_c))
        print('conv shape:', self.text_cov_list_c[0].shape)
        print('proto shape:', self.text_prototype_c.shape)

    def cmp(self, text_features,video_features, labels, labels_set): # class mean prototype
        class_label = []
        feature_dim = video_features.shape[1]

        text_prototype = []
        text_radius = []
        text_cov_list = []

        video_prototype = []
        video_radius = []
        video_cov_list = []

        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)

            text_feature_classwise = text_features[index]
            text_prototype.append(np.mean(text_feature_classwise, axis=0))
            text_cov = np.cov(text_feature_classwise.T)
            text_radius.append(np.sqrt(np.trace(text_cov) / feature_dim))
            text_cov_list.append(text_cov)

            video_feature_classwise = video_features[index]
            video_prototype.append(np.mean(video_feature_classwise, axis=0))
            video_cov = np.cov(video_feature_classwise.T)
            video_radius.append(np.sqrt(np.trace(video_cov) / feature_dim))
            video_cov_list.append(video_cov)

        return text_prototype, text_radius,  text_cov_list,  video_prototype, video_radius, video_cov_list, class_label

    
class Evaluator(Trainer):
    def __init__(self, model, metrics, config, eval_task_id, valid_data_loader, tokenizer, 
                 list_val_acc_ii, experiment=None):
        self.model = model
        self.metrics = metrics
        self.config = config
        self.task_id = eval_task_id
        self.val_loaders_list = valid_data_loader
        self.tokenizer = tokenizer
        self.list_val_acc_ii = list_val_acc_ii
        self.experiment = experiment
        
        # Initialize Accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Prepare the model with Accelerator
        self.model = self.accelerator.prepare(self.model)
        
        self.window_metric = defaultdict(list)
        self.checkpoint_dir = config.eval_path if hasattr(config, 'eval_path') and config.eval_path else config.model_path
        
        # Set up the logging files
        self.overall_log = os.path.join(self.checkpoint_dir, "overall_log.csv")
        log_num_tasks = int(getattr(config, "task_num", 20) or 20)
        if not os.path.exists(self.overall_log):
            from modules.trainer_utils import construct_exp_log
            construct_exp_log(self.overall_log, num_tasks=log_num_tasks)

        self.task_log = os.path.join(self.checkpoint_dir, "task_log.csv")
        if not os.path.exists(self.task_log):
            from modules.trainer_utils import construct_exp_log
            construct_exp_log(self.task_log, num_tasks=log_num_tasks)

        # Initialize the Validator object for validation
        self.validator = Validator(
            self.model, metrics, config, self.task_id, valid_data_loader, tokenizer, 
            self.accelerator, experiment, self.checkpoint_dir,
            self.list_val_acc_ii,  
            self.task_log, self.overall_log
        )
