import torch
import time
import os
from modules.metrics import text_embed_processing
from modules.trainer_utils import (
    log_validation_progress, log_validation_results,
    log_final_validation_progress, save_video_embeddings, load_stored_embed,
    save_task_prototype, AverageMeter, update_exp_result
)
import torch.nn.functional as F
from collections import defaultdict

def text_embed_processing_variable(text_embeds, all_vid_ids):
    """
    return: [(vid_id, tensor_of_text_embeds_for_that_vid), ...]
    """
    text_embeds_per_video_id = defaultdict(list)

    for embed, vid in zip(text_embeds, all_vid_ids):
        text_embeds_per_video_id[vid].append(embed)

    result = []
    for vid, embeds in text_embeds_per_video_id.items():
        embeds_tensor = torch.stack(embeds, dim=0)  # shape = (num_texts_for_vid, embed_dim)
        result.append((vid, embeds_tensor))

    return result  # list of (video_id, tensor)


import torch
import torch.nn.functional as F


def batch_word_frame_similarity_from_grouped_v3(
    text_embeds_per_video_id,
    vid_embeds,
    eps=1e-8,
    temp=0.07
):
    """
    - output shape: [num_texts, num_videos]
    - group-aware text flattening
    - per-text vs per-video soft alignment using max pooling
    - fully autograd-friendly
    """

    device = vid_embeds.device
    num_videos, N, D = vid_embeds.shape

    # Normalize video frame embeddings once
    vid_embeds = F.normalize(vid_embeds, dim=-1, eps=eps)  # [V, N, D]

    # ---- Flatten all texts (keeping group info) ----
    all_text_embeds = []
    text_to_video_idx = []

    for vid_idx, (_, text_list) in enumerate(text_embeds_per_video_id):
        for t in text_list:
            t = F.normalize(t.to(device), dim=-1, eps=eps)   # [L, D]
            all_text_embeds.append(t)
            text_to_video_idx.append(vid_idx)

    num_texts = len(all_text_embeds)

    # Prepare output
    sims = torch.zeros(num_texts, num_videos, device=device, dtype=vid_embeds.dtype)

    # ---- Compute similarities ----
    # No double-normalization and no redundant per-loop ops
    for i, t_embed in enumerate(all_text_embeds):
        if t_embed.numel() == 0:
            continue

        # t_embed: [L, D]
        # vid_embeds: [V, N, D]

        # Compute S for all videos at once using batch matmul
        # S: [V, N, L]
        S = torch.matmul(vid_embeds, t_embed.T)  # broadcasted batch matmul

        # max-over-frames (N-dim), then mean over words (L)
        max_over_frames = S.max(dim=1).values.mean(dim=1)   # [V]

        # max-over-words (L-dim), then mean over frames (N)
        max_over_words = S.max(dim=2).values.mean(dim=1)    # [V]

        sims[i] = 0.5 * (max_over_frames + max_over_words)

    return sims

from typing import List, Tuple

def is_variable_length_embeddings(arr):
        
        if not all(isinstance(x, torch.Tensor) and x.dim() == 2 for x in arr):
            return False
        token_lens = [x.shape[0] for x in arr]
        embed_dims = [x.shape[1] for x in arr]
        return len(set(token_lens)) > 1 and len(set(embed_dims)) == 1


import torch
import torch.nn.functional as F


class Validator:
    def __init__(self, model, metrics, config, task_id, val_loaders_list, tokenizer, accelerator, experiment, checkpoint_dir, list_val_acc_ii, task_log, overall_log):
        self.model = model
        self.metrics = metrics
        self.config = config
        self.task_id = task_id
        self.val_loaders_list = val_loaders_list
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.experiment = experiment
        self.checkpoint_dir = checkpoint_dir
        self.list_val_acc_ii = list_val_acc_ii
        self.task_log = task_log
        self.overall_log = overall_log
        self.device = accelerator.device

    def _v2_debug_enabled(self):
        return bool(getattr(self.config, 'debug_v2', False))

    def _v2_stats(self, tensor):
        if tensor is None or not torch.is_tensor(tensor):
            return 'None'
        with torch.no_grad():
            value = tensor.detach().float()
            return (
                f"shape={tuple(value.shape)}, min={value.min().item():.6f}, "
                f"max={value.max().item():.6f}, mean={value.mean().item():.6f}, "
                f"std={value.std(unbiased=False).item():.6f}"
            )

    def _v2_text_summary(self, text_embed_arr):
        if not text_embed_arr:
            return 'num_text=0'
        lengths, dims = [], []
        for item in text_embed_arr:
            if torch.is_tensor(item):
                lengths.append(item.shape[0] if item.dim() > 1 else 1)
                dims.append(item.shape[-1])
        if not lengths:
            return f"num_text={len(text_embed_arr)}"
        return (
            f"num_text={len(text_embed_arr)}, token_len_min={min(lengths)}, "
            f"token_len_max={max(lengths)}, token_len_mean={sum(lengths)/len(lengths):.2f}, "
            f"embed_dims={sorted(set(dims))}"
        )

    def _v2_print_validation_debug(self, tag, task_id, epoch, text_embed_arr, vid_embeds, sims, all_vid_ids):
        # Debug option: validation diagnostics are read-only and gated by config.debug_v2.
        if not self._v2_debug_enabled():
            return
        unique_vids = len(set(all_vid_ids)) if all_vid_ids is not None else 0
        print(f"\n[V2-VAL] tag={tag} task={task_id} epoch={epoch} {self._v2_text_summary(text_embed_arr)} unique_vids={unique_vids}")
        print(f"[V2-VAL] video={self._v2_stats(vid_embeds)}")
        print(f"[V2-VAL] sims={self._v2_stats(sims)}")

    def _feature_extraction(self, data, prototype_id=None):
        if self.tokenizer is not None:
            data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
        if isinstance(data['text'], torch.Tensor):
            data['text'] = data['text'].to(self.device)
        else:
            data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
        data['video'] = data['video'].to(self.device)
        if prototype_id is not None:
            data['prototype_id'] = prototype_id
        text_embed, vid_embed = self.model(data)
        return text_embed, vid_embed

    def _feature_extraction_wf(self, data, prototype_id=None):
        if self.tokenizer is not None:
            data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
        if isinstance(data['text'], torch.Tensor):
            data['text'] = data['text'].to(self.device)
        else:
            data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
        data['video'] = data['video'].to(self.device)
        if prototype_id is not None:
            data['prototype_id'] = prototype_id
        text_embed, vid_embed = self.model(data)
        return text_embed, vid_embed

    def _proto_feature_extraction(self, data, return_vid=False):
        if self.tokenizer is not None:
            data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
        if isinstance(data['text'], torch.Tensor):
            data['text'] = data['text'].to(self.device)
        else:
            data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
        data['video'] = data['video'].to(self.device)
        text_embed = self.model.forward_text(data['text']['input_ids'], data['text']['attention_mask'], self.task_id)
        if return_vid:    
            vid_embed = self.model.forward_video(data['video'])
            return text_embed, vid_embed
        return text_embed          

    def _proto_feature_extraction_wf(self, data, return_vid=False):
        if self.tokenizer is not None:
            data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
        if isinstance(data['text'], torch.Tensor):
            data['text'] = data['text'].to(self.device)
        else:
            data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
        data['video'] = data['video'].to(self.device)
        text_embed = self.model.forward_text_wf(data['text']['input_ids'], data['text']['attention_mask'], self.task_id)
        if return_vid:    
            vid_embed = self.model.forward_video_wf(data['video'])
            return text_embed, vid_embed
        return text_embed

    def task_validation_ori(self, task_id, epoch):
        """Validate on the current task."""
        self.model.eval()
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        validation_start_time = time.time()
        if epoch == 0:
            print("Initial Validation Start...")
        else:
            print("\nValidate on the current task...")
        with self.experiment.validate():
            with torch.no_grad():
                val_loader, num_classes = self.val_loaders_list[task_id - 1]
                prototype_id = task_id if self.config.task_prototype else None
                num_batches = len(val_loader)
                # Extract all video and text embeddings for the current task
                for batch_idx, data in enumerate(val_loader):
                    text_embed, vid_embed = self._feature_extraction(data, prototype_id=prototype_id)
                    text_embed_arr.append(text_embed)
                    vid_embed_arr.append(vid_embed)
                    for v_id in data['video_id']:
                        all_vid_ids.append(v_id)
                    log_validation_progress(epoch, batch_idx, num_batches, validation_start_time)
                text_embeds = torch.cat(text_embed_arr) # (num_samples, embed_dim)
                vid_embeds = torch.cat(vid_embed_arr) # (num_samples, embed_dim)
                
                # Remove duplicate videos if multiple captions per video (If one-to-many mapping)
                vid_embeds_per_video_id = {}
                for idx, v_id in enumerate(all_vid_ids):
                    if v_id not in vid_embeds_per_video_id:
                        vid_embeds_per_video_id[v_id] = vid_embeds[idx]
                vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

                # Process text embeddings for each video
                text_embeds_per_video_id = text_embed_processing(text_embeds, all_vid_ids, 1)

                # Calculate similarity scores and measure performance
                sims = text_embeds_per_video_id @ vid_embeds.t()
                res = self.metrics(sims)
                print(f"\n-----Task Val Epoch: {epoch}-----\n"
                      f"R@1: {res['R1']}\n"
                      f"R@5: {res['R5']}\n"
                      f"R@10: {res['R10']}\n"
                      f"MedR: {res['MedR']}\n"
                      f"MeanR: {res['MeanR']}")
                self.experiment.log_metric(f"R@1_of_Task{task_id}", res['R1'])

        return res


    def validate_ori(self, task_id, epoch, step):
        """General validation that considers all tasks."""
        self.model.eval()
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        curr_vid_ids = []
        num_batches = 0
        batch_indices = 0
        validation_start_time = time.time()
        print("\nValidate on all data...")
        with self.experiment.validate():
            with torch.no_grad():
                for n_task, (val_loader, num_classes) in enumerate(self.val_loaders_list):
                    num_batches += len(val_loader)
                    for data in val_loader:
                        if n_task == task_id - 1:
                            if self.config.task_prototype:
                                text_embed, vid_embed = self._proto_feature_extraction(data, return_vid=True)
                            else:
                                text_embed, vid_embed = self._feature_extraction(data)
                        else:
                            if self.config.task_prototype:
                                text_embed = self._proto_feature_extraction(data)
                            else:
                                text_embed, _ = self._feature_extraction(data)
                        text_embed_arr.append(text_embed)
                        if n_task == task_id - 1:
                            vid_embed_arr.append(vid_embed)
                            curr_vid_ids.extend(data['video_id'])
                        for v_id in data['video_id']:
                            all_vid_ids.append(v_id)
                        log_validation_progress(epoch, batch_indices, num_batches, validation_start_time)
                        batch_indices += 1

                if self.config.task_prototype:
                    text_embeds = torch.cat(text_embed_arr, dim=1)
                else:
                    text_embeds = torch.cat(text_embed_arr)

                if len(vid_embed_arr) == 0:
                    raise RuntimeError(f"vid_embed_arr is empty before torch.cat. This likely means no video embeddings were collected for the current task (n_task={n_task}, task_id={task_id}). Check the logic above for populating vid_embed_arr.")
                curr_vid_embeds = torch.cat(vid_embed_arr)
                # Remove duplicate video embeddings
                vid_embeds_per_video_id = {}
                for idx, v_id in enumerate(curr_vid_ids):
                    if v_id not in vid_embeds_per_video_id:
                        vid_embeds_per_video_id[v_id] = curr_vid_embeds[idx]
                curr_vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

                # Load stored video embeddings from previous tasks
                vid_embeds_arr = []
                if task_id > 1:
                    for n_task in range(task_id - 1):
                        task_vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
                        vid_embeds_arr.append(task_vid_embeds.to(self.device))
                vid_embeds_arr.append(curr_vid_embeds)

                if self.config.task_prototype:
                    total_sims = []
                    for prototype_id in range(task_id):
                        task_text_embeds = text_embeds[prototype_id]
                        text_embeds_per_video_id = text_embed_processing(task_text_embeds, all_vid_ids, 1)
                        task_sims = text_embeds_per_video_id @ vid_embeds_arr[prototype_id].t()
                        total_sims.append(task_sims)
                    sims = torch.cat(total_sims, dim=-1)
                else:
                    total_vid_embeds = torch.cat(vid_embeds_arr)
                    text_embeds_per_video_id = text_embed_processing(text_embeds, all_vid_ids, 1)
                    sims = text_embeds_per_video_id @ total_vid_embeds.t()
                res = self.metrics(sims)
                log_validation_results(self.experiment, task_id, step, epoch, res, self.config)
        return res

    def final_validate_ori(self, task_id, list_val_acc_ii):
        """Final evaluation at the end of training for all tasks."""
        if self.config.load_best:
            checkpoint_path = f'task{task_id}_model_best.pth'
            self._load_checkpoint(checkpoint_path)
        self.model.eval()
        BWF = AverageMeter()
        validation_start_time = time.time()
        total_tasks = len(self.val_loaders_list)

        print("\nFinal Validation Start...")
        with self.experiment.validate():
            with torch.no_grad():
                for n_task, (val_loader, num_classes) in enumerate(self.val_loaders_list):
                    task_start_time = time.time()
                    prototype_id = n_task + 1 if self.config.task_prototype else None
                    num_batches = len(val_loader)
                    text_embed_arr = []
                    vid_embed_arr = []
                    all_vid_ids = []
                    print(f"Task: {n_task + 1}/{total_tasks}")
                    for batch_idx, data in enumerate(val_loader):
                        text_embed, vid_embed = self._feature_extraction(data, prototype_id)
                        text_embed_arr.append(text_embed)
                        vid_embed_arr.append(vid_embed)
                        for v_id in data['video_id']:
                            all_vid_ids.append(v_id)
                        log_final_validation_progress(n_task, total_tasks, batch_idx, num_batches, task_start_time, validation_start_time)
                    text_embeds = torch.cat(text_embed_arr)
                    vid_embeds = torch.cat(vid_embed_arr)
                    # Remove duplicate video embeddings
                    vid_embeds_per_video_id = {}
                    for idx, v_id in enumerate(all_vid_ids):
                        if v_id not in vid_embeds_per_video_id:
                            vid_embeds_per_video_id[v_id] = vid_embeds[idx]
                    vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
                    if n_task < task_id - 1:
                        vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
                    else:
                        save_video_embeddings(self.checkpoint_dir, n_task + 1, vid_embeds)
                        if self.config.task_prototype:
                            prototype = self.model.clipmodel.text_model.task_prototype[n_task]
                            save_task_prototype(self.checkpoint_dir, n_task + 1, prototype)
                    text_embeds_per_video_id = text_embed_processing(text_embeds, all_vid_ids, 1)
                    sims = text_embeds_per_video_id @ vid_embeds.t()
                    res = self.metrics(sims)
                    if n_task == task_id - 1:
                        list_val_acc_ii.append(res['R1'])
                        print(f"\nValidation R@1 List: {list_val_acc_ii}")
                        print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                    elif n_task < task_id - 1:
                        if len(list_val_acc_ii) == 0:
                            print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                        else:
                            forgetting = list_val_acc_ii[n_task] - res['R1']
                            print(f"\nValidation R@1 List: {list_val_acc_ii}")
                            print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                            print(f"Task {n_task + 1} Forgetting: {forgetting:.6f}")
                            BWF.update(forgetting, num_classes)
                            print(f"Task {n_task + 1} BWF: {BWF.avg:.6f}")
        update_exp_result(self.task_log, task_id, bwf=BWF.avg)
        update_exp_result(self.overall_log, task_id, bwf=BWF.avg)
        if task_id == 1 and self.config.task_prototype:
            self._duplicate_weights()
            self._save_checkpoint(0, save_best=True)
        return BWF.avg


    def _normalize_text_embed_output_v2(self, text_embed):
        """
        Normalize model text outputs into a list while preserving global batch
        embeddings as batch tensors and word-frame embeddings as per-sample
        [num_words, dim] tensors.
        """
        if isinstance(text_embed, torch.Tensor):
            if text_embed.dim() == 3:
                return [sample for sample in text_embed]
            return [text_embed]
        if isinstance(text_embed, list):
            outputs = []
            for item in text_embed:
                outputs.extend(self._normalize_text_embed_output_v2(item))
            return outputs
        raise TypeError(f"Unexpected type for text_embed: {type(text_embed)}")

    def _is_word_frame_text_embeds_v2(self, text_embed_arr, all_vid_ids):
        """
        Detect word-frame text embeddings by structure instead of variable token
        length. This keeps the fine-grained path correct even when all captions
        have the same number of tokens.
        """
        if len(text_embed_arr) == 0 or len(text_embed_arr) != len(all_vid_ids):
            return False
        if not all(isinstance(x, torch.Tensor) and x.dim() == 2 for x in text_embed_arr):
            return False
        # If every item has a single row, this is very likely batch-size-1 global
        # embeddings rather than token-level word embeddings.
        return any(x.shape[0] > 1 for x in text_embed_arr)

    def _deduplicate_video_embeds_v2(self, vid_embeds, video_ids):
        vid_embeds_per_video_id = {}
        for idx, v_id in enumerate(video_ids):
            if v_id not in vid_embeds_per_video_id:
                vid_embeds_per_video_id[v_id] = vid_embeds[idx]
        return torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

    def _compute_similarity_v2(self, text_embed_arr, vid_embeds, all_vid_ids):
        """
        Compute retrieval similarities for either global embeddings or the
        fine-grained word-frame similarity:

            0.5 * (mean_words max_frames <w, f> + mean_frames max_words <w, f>)
        """
        if self._is_word_frame_text_embeds_v2(text_embed_arr, all_vid_ids):
            text_embeds_per_video_id = text_embed_processing_variable(
                text_embed_arr, all_vid_ids
            )
            sims = batch_word_frame_similarity_from_grouped_v3(
                text_embeds_per_video_id, vid_embeds, eps=1e-8
            )
            return sims.unsqueeze(1)

        text_embeds = torch.cat(text_embed_arr)
        text_embeds_per_video_id = text_embed_processing(
            text_embeds, all_vid_ids, 1
        )
        return text_embeds_per_video_id @ vid_embeds.t()

    def _fusion_v3_enabled(self):
        # Inference fusion switch:
        # false -> original word-frame evaluation path
        # true  -> fused word-frame + global similarity evaluation path
        return bool(getattr(self.config, 'use_fusion_v3', False))

    def _fusion_v3_alpha(self):
        return float(getattr(self.config, 'fusion_v3_alpha', 0.5))

    def _row_zscore_v3(self, sims, eps=1e-6):
        mean = sims.mean(dim=-1, keepdim=True)
        std = sims.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (sims - mean) / std

    def _global_video_from_frames_v3(self, vid_embeds, eps=1e-8):
        # Fusion note: this matches the offline alpha=0.5 ablation, using mean-pooled frame embeddings.
        if vid_embeds.dim() == 3:
            vid_embeds = vid_embeds.mean(dim=1)
        return F.normalize(vid_embeds, dim=-1, eps=eps)

    def _compute_global_similarity_v3(self, text_embed_arr, vid_embeds, all_vid_ids):
        if len(text_embed_arr) == 0:
            raise RuntimeError("No global text embeddings available for fusion_v3.")
        text_embeds = torch.cat(text_embed_arr)
        text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-8)
        text_embeds_per_video_id = text_embed_processing(text_embeds, all_vid_ids, 1)
        global_vid_embeds = self._global_video_from_frames_v3(vid_embeds)
        return text_embeds_per_video_id @ global_vid_embeds.t()

    def _compute_similarity_parts_v3(self, text_wf_arr, vid_embeds, all_vid_ids, text_global_arr=None):
        """Return unfused word-frame and global similarities for full-matrix V3 fusion."""
        wf_sims = self._compute_similarity_v2(text_wf_arr, vid_embeds, all_vid_ids)
        if text_global_arr is None or len(text_global_arr) == 0:
            return wf_sims, None
        global_sims = self._compute_global_similarity_v3(text_global_arr, vid_embeds, all_vid_ids)
        return wf_sims, global_sims

    def _fuse_similarity_v3(self, wf_sims, global_sims):
        # Fusion note: this reproduces the offline ablation: fuse after the full similarity matrix is built.
        if not self._fusion_v3_enabled() or global_sims is None:
            if global_sims is None and self._fusion_v3_enabled():
                print("[V3-VAL] Missing global text embeddings; falling back to v2 word-frame similarity.")
            return wf_sims
        alpha = self._fusion_v3_alpha()
        fused = alpha * self._row_zscore_v3(wf_sims) + (1.0 - alpha) * self._row_zscore_v3(global_sims)
        if self._v2_debug_enabled():
            print(
                f"[V3-VAL] full-matrix fusion alpha={alpha:.3f} wf={self._v2_stats(wf_sims)} "
                f"global={self._v2_stats(global_sims)} fused={self._v2_stats(fused)}"
            )
        return fused

    def _compute_similarity_v3(self, text_wf_arr, vid_embeds, all_vid_ids, text_global_arr=None):
        """V3 score fusion: alpha*zscore(word-frame) + (1-alpha)*zscore(global)."""
        wf_sims, global_sims = self._compute_similarity_parts_v3(
            text_wf_arr, vid_embeds, all_vid_ids, text_global_arr
        )
        return self._fuse_similarity_v3(wf_sims, global_sims)

    def _feature_extraction_fusion_v3(self, data, prototype_id=None, proto_text=False, return_vid=True):
        # Fusion note: helper is additive; v2 feature extraction remains untouched.
        if self.tokenizer is not None and not isinstance(data['text'], dict):
            data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
        if isinstance(data['text'], torch.Tensor):
            data['text'] = data['text'].to(self.device)
        else:
            data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
        data['video'] = data['video'].to(self.device)

        if proto_text:
            text_wf = self.model.forward_text_wf(
                data['text']['input_ids'], data['text']['attention_mask'], self.task_id
            )
            text_global = self.model.forward_text(
                data['text']['input_ids'], data['text']['attention_mask'], self.task_id
            )
            if return_vid:
                vid_embed = self.model.forward_video_wf(data['video'])
                return text_wf, text_global, vid_embed
            return text_wf, text_global

        if prototype_id is not None:
            data['prototype_id'] = prototype_id
        text_wf, vid_embed = self.model(data)
        global_proto = prototype_id if prototype_id is not None else self.task_id
        text_global = self.model.forward_text(
            data['text']['input_ids'], data['text']['attention_mask'], global_proto
        )
        return text_wf, text_global, vid_embed

    def _append_text_embeds_by_proto_v2(self, text_embeds_by_proto, text_embed, task_id):
        if isinstance(text_embed, list):
            if len(text_embed) < task_id:
                raise RuntimeError(
                    f"Expected at least {task_id} prototype text outputs, got {len(text_embed)}"
                )
            for prototype_id in range(task_id):
                text_embeds_by_proto[prototype_id].extend(
                    self._normalize_text_embed_output_v2(text_embed[prototype_id])
                )
            return True
        return False

    def task_validation_v2(self, task_id, epoch):
        """Validate on the current task with shared word-frame similarity logic."""
        self.model.eval()
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        validation_start_time = time.time()
        if epoch == 0:
            print("Initial Validation Start...")
        else:
            print("\nValidate on the current task...")
        with self.experiment.validate():
            with torch.no_grad():
                val_loader, num_classes = self.val_loaders_list[task_id - 1]
                prototype_id = task_id if self.config.task_prototype else None
                num_batches = len(val_loader)
                for batch_idx, data in enumerate(val_loader):
                    text_embed, vid_embed = self._feature_extraction(data, prototype_id=prototype_id)
                    text_embed_arr.extend(self._normalize_text_embed_output_v2(text_embed))
                    vid_embed_arr.append(vid_embed)
                    all_vid_ids.extend(data['video_id'])
                    log_validation_progress(epoch, batch_idx, num_batches, validation_start_time)

                vid_embeds = torch.cat(vid_embed_arr)
                vid_embeds = self._deduplicate_video_embeds_v2(vid_embeds, all_vid_ids)
                sims = self._compute_similarity_v2(text_embed_arr, vid_embeds, all_vid_ids)
                self._v2_print_validation_debug('task_validation_v2', task_id, epoch, text_embed_arr, vid_embeds, sims, all_vid_ids)
                res = self.metrics(sims)
                print(f"\n-----Task Val Epoch: {epoch}-----\n"
                      f"R@1: {res['R1']}\n"
                      f"R@5: {res['R5']}\n"
                      f"R@10: {res['R10']}\n"
                      f"MedR: {res['MedR']}\n"
                      f"MeanR: {res['MeanR']}")
                self.experiment.log_metric(f"R@1_of_Task{task_id}", res['R1'])
        return res

    def validate_v2(self, task_id, epoch, step):
        """General validation over all seen tasks using shared similarity logic."""
        self.model.eval()
        text_embeds_by_proto = [[] for _ in range(task_id)] if self.config.task_prototype else [[]]
        proto_tensor_text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        curr_vid_ids = []
        num_batches = 0
        batch_indices = 0
        validation_start_time = time.time()
        print("\nValidate on all data...")
        with self.experiment.validate():
            with torch.no_grad():
                for n_task, (val_loader, num_classes) in enumerate(self.val_loaders_list):
                    num_batches += len(val_loader)
                    for data in val_loader:
                        if n_task == task_id - 1:
                            if self.config.task_prototype:
                                text_embed, vid_embed = self._proto_feature_extraction_wf(data, return_vid=True)
                            else:
                                text_embed, vid_embed = self._feature_extraction(data)
                        else:
                            if self.config.task_prototype:
                                text_embed = self._proto_feature_extraction_wf(data)
                            else:
                                text_embed, _ = self._feature_extraction(data)

                        if self.config.task_prototype:
                            handled = self._append_text_embeds_by_proto_v2(
                                text_embeds_by_proto, text_embed, task_id
                            )
                            if not handled:
                                proto_tensor_text_embed_arr.append(text_embed)
                        else:
                            text_embeds_by_proto[0].extend(
                                self._normalize_text_embed_output_v2(text_embed)
                            )

                        if n_task == task_id - 1:
                            vid_embed_arr.append(vid_embed)
                            curr_vid_ids.extend(data['video_id'])
                        all_vid_ids.extend(data['video_id'])
                        log_validation_progress(epoch, batch_indices, num_batches, validation_start_time)
                        batch_indices += 1

                if self.config.task_prototype and len(proto_tensor_text_embed_arr) > 0:
                    text_embeds = torch.cat(proto_tensor_text_embed_arr, dim=1)
                    for prototype_id in range(task_id):
                        text_embeds_by_proto[prototype_id] = [text_embeds[prototype_id]]

                if len(vid_embed_arr) == 0:
                    raise RuntimeError(
                        "vid_embed_arr is empty before torch.cat. This likely means no video "
                        "embeddings were collected for the current task."
                    )
                curr_vid_embeds = torch.cat(vid_embed_arr)
                curr_vid_embeds = self._deduplicate_video_embeds_v2(curr_vid_embeds, curr_vid_ids)

                vid_embeds_arr = []
                if task_id > 1:
                    for n_task in range(task_id - 1):
                        task_vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
                        if self._v2_debug_enabled():
                            print(f"[V2-LOAD] validate_v2 loaded task={n_task + 1} video={self._v2_stats(task_vid_embeds)}")
                        vid_embeds_arr.append(task_vid_embeds.to(self.device))
                vid_embeds_arr.append(curr_vid_embeds)

                if self.config.task_prototype:
                    total_sims = []
                    for prototype_id in range(task_id):
                        task_sims = self._compute_similarity_v2(
                            text_embeds_by_proto[prototype_id],
                            vid_embeds_arr[prototype_id],
                            all_vid_ids
                        )
                        total_sims.append(task_sims)
                    sims = torch.cat(total_sims, dim=-1)
                    if self._v2_debug_enabled():
                        print(f"[V2-VAL] validate_v2 prototype_parts={len(total_sims)} total_sims={self._v2_stats(sims)}")
                else:
                    total_vid_embeds = torch.cat(vid_embeds_arr)
                    sims = self._compute_similarity_v2(
                        text_embeds_by_proto[0], total_vid_embeds, all_vid_ids
                    )

                flat_text_arr = []
                for arr in text_embeds_by_proto:
                    flat_text_arr.extend(arr)
                self._v2_print_validation_debug('validate_v2', task_id, epoch, flat_text_arr, torch.cat(vid_embeds_arr), sims, all_vid_ids)
                res = self.metrics(sims)
                log_validation_results(self.experiment, task_id, step, epoch, res, self.config)
        return res

    def final_validate_v2(self, task_id, list_val_acc_ii):
        """Final evaluation with shared word-frame similarity logic."""
        if self.config.load_best:
            checkpoint_path = f'task{task_id}_model_best.pth'
            self._load_checkpoint(checkpoint_path)
        self.model.eval()
        BWF = AverageMeter()
        validation_start_time = time.time()
        total_tasks = len(self.val_loaders_list)

        print("\nFinal Validation Start...")
        with self.experiment.validate():
            with torch.no_grad():
                for n_task, (val_loader, num_classes) in enumerate(self.val_loaders_list):
                    task_start_time = time.time()
                    prototype_id = n_task + 1 if self.config.task_prototype else None
                    num_batches = len(val_loader)
                    text_embed_arr = []
                    vid_embed_arr = []
                    all_vid_ids = []
                    print(f"Task: {n_task + 1}/{total_tasks}")
                    for batch_idx, data in enumerate(val_loader):
                        text_embed, vid_embed = self._feature_extraction(data, prototype_id)
                        text_embed_arr.extend(self._normalize_text_embed_output_v2(text_embed))
                        vid_embed_arr.append(vid_embed)
                        all_vid_ids.extend(data['video_id'])
                        log_final_validation_progress(
                            n_task, total_tasks, batch_idx, num_batches,
                            task_start_time, validation_start_time
                        )

                    vid_embeds = torch.cat(vid_embed_arr)
                    vid_embeds = self._deduplicate_video_embeds_v2(vid_embeds, all_vid_ids)
                    if n_task < task_id - 1:
                        vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
                        if self._v2_debug_enabled():
                            print(f"[V2-LOAD] final_validate_v2 loaded task={n_task + 1} video={self._v2_stats(vid_embeds)}")
                    else:
                        save_video_embeddings(self.checkpoint_dir, n_task + 1, vid_embeds)
                        if self.config.task_prototype:
                            prototype = self.model.clipmodel.text_model.task_prototype[n_task]
                            save_task_prototype(self.checkpoint_dir, n_task + 1, prototype)

                    sims = self._compute_similarity_v2(text_embed_arr, vid_embeds, all_vid_ids)
                    self._v2_print_validation_debug('final_validate_v2', n_task + 1, task_id, text_embed_arr, vid_embeds, sims, all_vid_ids)
                    res = self.metrics(sims)
                    if n_task == task_id - 1:
                        list_val_acc_ii.append(res['R1'])
                        print(f"\nValidation R@1 List: {list_val_acc_ii}")
                        print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                    elif n_task < task_id - 1:
                        if len(list_val_acc_ii) == 0:
                            print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                        else:
                            forgetting = list_val_acc_ii[n_task] - res['R1']
                            print(f"\nValidation R@1 List: {list_val_acc_ii}")
                            print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                            print(f"Task {n_task + 1} Forgetting: {forgetting:.6f}")
                            BWF.update(forgetting, num_classes)
                            print(f"Task {n_task + 1} BWF: {BWF.avg:.6f}")
        update_exp_result(self.task_log, task_id, bwf=BWF.avg)
        update_exp_result(self.overall_log, task_id, bwf=BWF.avg)
        if task_id == 1 and self.config.task_prototype:
            self._duplicate_weights()
            self._save_checkpoint(0, save_best=True)
        return BWF.avg

    def task_validation_v3(self, task_id, epoch):
        """Validate the current task with V3 fusion aligned to prototype-aware full-matrix logic."""
        self.model.eval()
        text_wf_by_proto = [[] for _ in range(task_id)] if self.config.task_prototype else [[]]
        text_global_by_proto = [[] for _ in range(task_id)] if self.config.task_prototype else [[]]
        proto_tensor_text_wf_arr = []
        proto_tensor_text_global_arr = []
        vid_embed_arr = []
        curr_vid_ids = []
        validation_start_time = time.time()
        if epoch == 0:
            print("Initial Validation Start...")
        else:
            print()
            print("Validate on the current task with V3 fusion...")
        with self.experiment.validate():
            with torch.no_grad():
                val_loader, num_classes = self.val_loaders_list[task_id - 1]
                num_batches = len(val_loader)
                for batch_idx, data in enumerate(val_loader):
                    if self.config.task_prototype:
                        text_wf, text_global, vid_embed = self._feature_extraction_fusion_v3(
                            data, proto_text=True, return_vid=True
                        )
                        handled_wf = self._append_text_embeds_by_proto_v2(text_wf_by_proto, text_wf, task_id)
                        handled_global = self._append_text_embeds_by_proto_v2(text_global_by_proto, text_global, task_id)
                        if not handled_wf:
                            proto_tensor_text_wf_arr.append(text_wf)
                        if not handled_global:
                            proto_tensor_text_global_arr.append(text_global)
                    else:
                        prototype_id = task_id if self.config.task_prototype else None
                        text_wf, text_global, vid_embed = self._feature_extraction_fusion_v3(
                            data, prototype_id=prototype_id
                        )
                        text_wf_by_proto[0].extend(self._normalize_text_embed_output_v2(text_wf))
                        text_global_by_proto[0].extend(self._normalize_text_embed_output_v2(text_global))
                    vid_embed_arr.append(vid_embed)
                    curr_vid_ids.extend(data['video_id'])
                    log_validation_progress(epoch, batch_idx, num_batches, validation_start_time)

                if self.config.task_prototype and len(proto_tensor_text_wf_arr) > 0:
                    text_embeds = torch.cat(proto_tensor_text_wf_arr, dim=1)
                    for prototype_id in range(task_id):
                        text_wf_by_proto[prototype_id] = [text_embeds[prototype_id]]
                if self.config.task_prototype and len(proto_tensor_text_global_arr) > 0:
                    text_embeds = torch.cat(proto_tensor_text_global_arr, dim=1)
                    for prototype_id in range(task_id):
                        text_global_by_proto[prototype_id] = [text_embeds[prototype_id]]

                curr_vid_embeds = torch.cat(vid_embed_arr)
                curr_vid_embeds = self._deduplicate_video_embeds_v2(curr_vid_embeds, curr_vid_ids)

                vid_embeds_arr = []
                if task_id > 1:
                    for n_task in range(task_id - 1):
                        task_vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
                        if self._v2_debug_enabled():
                            print(
                                f"[V3-TASK-LOAD] task_validation_v3 loaded task={n_task + 1} "
                                f"video={self._v2_stats(task_vid_embeds)}"
                            )
                        vid_embeds_arr.append(task_vid_embeds.to(self.device))
                vid_embeds_arr.append(curr_vid_embeds)
                total_vid_embeds = torch.cat(vid_embeds_arr)

                if self.config.task_prototype:
                    wf_parts = []
                    global_parts = []
                    has_global_parts = True
                    for prototype_id in range(task_id):
                        wf_part, global_part = self._compute_similarity_parts_v3(
                            text_wf_by_proto[prototype_id],
                            vid_embeds_arr[prototype_id],
                            curr_vid_ids,
                            text_global_by_proto[prototype_id],
                        )
                        wf_parts.append(wf_part)
                        if global_part is None:
                            has_global_parts = False
                        else:
                            global_parts.append(global_part)
                    wf_sims = torch.cat(wf_parts, dim=-1)
                    global_sims = torch.cat(global_parts, dim=-1) if has_global_parts else None
                    fused_sims = self._fuse_similarity_v3(wf_sims, global_sims)
                else:
                    fused_sims = self._compute_similarity_v3(
                        text_wf_by_proto[0],
                        total_vid_embeds,
                        curr_vid_ids,
                        text_global_by_proto[0],
                    )

                curr_block_start = total_vid_embeds.shape[0] - curr_vid_embeds.shape[0]
                sims = fused_sims[..., curr_block_start:]

                flat_text_arr = []
                for arr in text_wf_by_proto:
                    flat_text_arr.extend(arr)
                self._v2_print_validation_debug(
                    'task_validation_v3',
                    task_id,
                    epoch,
                    flat_text_arr,
                    curr_vid_embeds,
                    sims,
                    curr_vid_ids,
                )
                if self._v2_debug_enabled():
                    print(
                        f"[V3-TASK] task_validation_v3 total_video={total_vid_embeds.shape[0]} "
                        f"current_video={curr_vid_embeds.shape[0]} curr_block_start={curr_block_start} "
                        f"fused_sims={self._v2_stats(fused_sims)} task_sims={self._v2_stats(sims)}"
                    )
                res = self.metrics(sims)
                print()
                print(f"-----Task Val V3 Epoch: {epoch}-----")
                print(f"R@1: {res['R1']}")
                print(f"R@5: {res['R5']}")
                print(f"R@10: {res['R10']}")
                print(f"MedR: {res['MedR']}")
                print(f"MeanR: {res['MeanR']}")
                self.experiment.log_metric(f"R@1_of_Task{task_id}_v3", res['R1'])
        return res

    def validate_v3(self, task_id, epoch, step):
        """General validation over all seen tasks with V3 word-frame/global fusion."""
        self.model.eval()
        text_wf_by_proto = [[] for _ in range(task_id)] if self.config.task_prototype else [[]]
        text_global_by_proto = [[] for _ in range(task_id)] if self.config.task_prototype else [[]]
        proto_tensor_text_wf_arr = []
        proto_tensor_text_global_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        curr_vid_ids = []
        num_batches = 0
        batch_indices = 0
        validation_start_time = time.time()
        print("\nValidate on all data with V3 fusion...")
        with self.experiment.validate():
            with torch.no_grad():
                for n_task, (val_loader, num_classes) in enumerate(self.val_loaders_list):
                    num_batches += len(val_loader)
                    for data in val_loader:
                        if n_task == task_id - 1:
                            if self.config.task_prototype:
                                text_wf, text_global, vid_embed = self._feature_extraction_fusion_v3(
                                    data, proto_text=True, return_vid=True
                                )
                            else:
                                text_wf, text_global, vid_embed = self._feature_extraction_fusion_v3(data)
                        else:
                            if self.config.task_prototype:
                                text_wf, text_global = self._feature_extraction_fusion_v3(
                                    data, proto_text=True, return_vid=False
                                )
                            else:
                                text_wf, text_global, _ = self._feature_extraction_fusion_v3(data)

                        if self.config.task_prototype:
                            handled_wf = self._append_text_embeds_by_proto_v2(text_wf_by_proto, text_wf, task_id)
                            handled_global = self._append_text_embeds_by_proto_v2(text_global_by_proto, text_global, task_id)
                            if not handled_wf:
                                proto_tensor_text_wf_arr.append(text_wf)
                            if not handled_global:
                                proto_tensor_text_global_arr.append(text_global)
                        else:
                            text_wf_by_proto[0].extend(self._normalize_text_embed_output_v2(text_wf))
                            text_global_by_proto[0].extend(self._normalize_text_embed_output_v2(text_global))

                        if n_task == task_id - 1:
                            vid_embed_arr.append(vid_embed)
                            curr_vid_ids.extend(data['video_id'])
                        all_vid_ids.extend(data['video_id'])
                        log_validation_progress(epoch, batch_indices, num_batches, validation_start_time)
                        batch_indices += 1

                if self.config.task_prototype and len(proto_tensor_text_wf_arr) > 0:
                    text_embeds = torch.cat(proto_tensor_text_wf_arr, dim=1)
                    for prototype_id in range(task_id):
                        text_wf_by_proto[prototype_id] = [text_embeds[prototype_id]]
                if self.config.task_prototype and len(proto_tensor_text_global_arr) > 0:
                    text_embeds = torch.cat(proto_tensor_text_global_arr, dim=1)
                    for prototype_id in range(task_id):
                        text_global_by_proto[prototype_id] = [text_embeds[prototype_id]]

                if len(vid_embed_arr) == 0:
                    raise RuntimeError("vid_embed_arr is empty before torch.cat in validate_v3.")
                curr_vid_embeds = torch.cat(vid_embed_arr)
                curr_vid_embeds = self._deduplicate_video_embeds_v2(curr_vid_embeds, curr_vid_ids)

                vid_embeds_arr = []
                if task_id > 1:
                    for n_task in range(task_id - 1):
                        task_vid_embeds = load_stored_embed(self.checkpoint_dir, n_task + 1)
                        if self._v2_debug_enabled():
                            print(f"[V3-LOAD] validate_v3 loaded task={n_task + 1} video={self._v2_stats(task_vid_embeds)}")
                        vid_embeds_arr.append(task_vid_embeds.to(self.device))
                vid_embeds_arr.append(curr_vid_embeds)

                if self.config.task_prototype:
                    # Fusion note: build full wf/global matrices first, then fuse once.
                    # This matches the offline fusion script and avoids per-task-block zscore calibration.
                    wf_parts = []
                    global_parts = []
                    has_global_parts = True
                    for prototype_id in range(task_id):
                        wf_part, global_part = self._compute_similarity_parts_v3(
                            text_wf_by_proto[prototype_id],
                            vid_embeds_arr[prototype_id],
                            all_vid_ids,
                            text_global_by_proto[prototype_id]
                        )
                        wf_parts.append(wf_part)
                        if global_part is None:
                            has_global_parts = False
                        else:
                            global_parts.append(global_part)
                    wf_sims = torch.cat(wf_parts, dim=-1)
                    global_sims = torch.cat(global_parts, dim=-1) if has_global_parts else None
                    sims = self._fuse_similarity_v3(wf_sims, global_sims)
                    if self._v2_debug_enabled():
                        print(f"[V3-VAL] validate_v3 prototype_parts={len(wf_parts)} full_matrix_sims={self._v2_stats(sims)}")
                else:
                    total_vid_embeds = torch.cat(vid_embeds_arr)
                    sims = self._compute_similarity_v3(
                        text_wf_by_proto[0], total_vid_embeds, all_vid_ids, text_global_by_proto[0]
                    )

                flat_text_arr = []
                for arr in text_wf_by_proto:
                    flat_text_arr.extend(arr)
                self._v2_print_validation_debug('validate_v3', task_id, epoch, flat_text_arr, torch.cat(vid_embeds_arr), sims, all_vid_ids)
                res = self.metrics(sims)
                log_validation_results(self.experiment, task_id, step, epoch, res, self.config)
        return res

    def final_validate_v3(self, task_id, list_val_acc_ii):
        """Final evaluation with V3 fusion aligned to prototype-aware full-matrix logic."""
        if self.config.load_best:
            checkpoint_path = f'task{task_id}_model_best.pth'
            self._load_checkpoint(checkpoint_path)
        self.model.eval()
        BWF = AverageMeter()
        validation_start_time = time.time()
        total_tasks = len(self.val_loaders_list)

        print()
        print("Final Validation V3 Start...")
        with self.experiment.validate():
            with torch.no_grad():
                # V3 fix note: precompute and cache the current-task video block first,
                # so earlier tasks can build the same full seen-task matrix without
                # depending on a not-yet-written task{task_id}_vid_embed.pth file.
                current_task_vid_embeds = None
                if task_id >= 1:
                    current_loader, _ = self.val_loaders_list[task_id - 1]
                    current_vid_embed_arr = []
                    current_vid_ids = []
                    for data in current_loader:
                        _, _, vid_embed = self._feature_extraction_fusion_v3(
                            data, proto_text=True, return_vid=True
                        ) if self.config.task_prototype else self._feature_extraction_fusion_v3(
                            data, prototype_id=task_id
                        )
                        current_vid_embed_arr.append(vid_embed)
                        current_vid_ids.extend(data['video_id'])
                    current_task_vid_embeds = torch.cat(current_vid_embed_arr)
                    current_task_vid_embeds = self._deduplicate_video_embeds_v2(
                        current_task_vid_embeds, current_vid_ids
                    )
                    save_video_embeddings(self.checkpoint_dir, task_id, current_task_vid_embeds)
                    if self.config.task_prototype:
                        prototype = self.model.clipmodel.text_model.task_prototype[task_id - 1]
                        save_task_prototype(self.checkpoint_dir, task_id, prototype)
                    if self._v2_debug_enabled():
                        print(
                            f"[V3-PRELOAD] final_validate_v3 prepared task={task_id} "
                            f"video={self._v2_stats(current_task_vid_embeds)}"
                        )
                for n_task, (val_loader, num_classes) in enumerate(self.val_loaders_list):
                    task_start_time = time.time()
                    num_batches = len(val_loader)
                    text_wf_by_proto = [[] for _ in range(task_id)] if self.config.task_prototype else [[]]
                    text_global_by_proto = [[] for _ in range(task_id)] if self.config.task_prototype else [[]]
                    proto_tensor_text_wf_arr = []
                    proto_tensor_text_global_arr = []
                    vid_embed_arr = []
                    eval_vid_ids = []
                    print(f"Task: {n_task + 1}/{total_tasks}")
                    for batch_idx, data in enumerate(val_loader):
                        if self.config.task_prototype:
                            text_wf, text_global, vid_embed = self._feature_extraction_fusion_v3(
                                data, proto_text=True, return_vid=True
                            )
                            handled_wf = self._append_text_embeds_by_proto_v2(text_wf_by_proto, text_wf, task_id)
                            handled_global = self._append_text_embeds_by_proto_v2(text_global_by_proto, text_global, task_id)
                            if not handled_wf:
                                proto_tensor_text_wf_arr.append(text_wf)
                            if not handled_global:
                                proto_tensor_text_global_arr.append(text_global)
                        else:
                            prototype_id = n_task + 1 if self.config.task_prototype else None
                            text_wf, text_global, vid_embed = self._feature_extraction_fusion_v3(
                                data, prototype_id=prototype_id
                            )
                            text_wf_by_proto[0].extend(self._normalize_text_embed_output_v2(text_wf))
                            text_global_by_proto[0].extend(self._normalize_text_embed_output_v2(text_global))
                        vid_embed_arr.append(vid_embed)
                        eval_vid_ids.extend(data['video_id'])
                        log_final_validation_progress(
                            n_task, total_tasks, batch_idx, num_batches,
                            task_start_time, validation_start_time
                        )

                    if self.config.task_prototype and len(proto_tensor_text_wf_arr) > 0:
                        text_embeds = torch.cat(proto_tensor_text_wf_arr, dim=1)
                        for prototype_id in range(task_id):
                            text_wf_by_proto[prototype_id] = [text_embeds[prototype_id]]
                    if self.config.task_prototype and len(proto_tensor_text_global_arr) > 0:
                        text_embeds = torch.cat(proto_tensor_text_global_arr, dim=1)
                        for prototype_id in range(task_id):
                            text_global_by_proto[prototype_id] = [text_embeds[prototype_id]]

                    eval_vid_embeds = torch.cat(vid_embed_arr)
                    eval_vid_embeds = self._deduplicate_video_embeds_v2(eval_vid_embeds, eval_vid_ids)
                    if n_task == task_id - 1:
                        eval_vid_embeds = current_task_vid_embeds if current_task_vid_embeds is not None else eval_vid_embeds

                    vid_embeds_arr = []
                    for seen_task in range(task_id):
                        if seen_task == n_task:
                            vid_block = eval_vid_embeds
                        elif seen_task == task_id - 1 and current_task_vid_embeds is not None:
                            vid_block = current_task_vid_embeds
                        else:
                            vid_block = load_stored_embed(self.checkpoint_dir, seen_task + 1).to(self.device)
                            if self._v2_debug_enabled():
                                print(
                                    f"[V3-LOAD] final_validate_v3 loaded task={seen_task + 1} "
                                    f"video={self._v2_stats(vid_block)}"
                                )
                        vid_embeds_arr.append(vid_block)
                    total_vid_embeds = torch.cat(vid_embeds_arr)

                    if self.config.task_prototype:
                        wf_parts = []
                        global_parts = []
                        has_global_parts = True
                        for prototype_id in range(task_id):
                            wf_part, global_part = self._compute_similarity_parts_v3(
                                text_wf_by_proto[prototype_id],
                                vid_embeds_arr[prototype_id],
                                eval_vid_ids,
                                text_global_by_proto[prototype_id],
                            )
                            wf_parts.append(wf_part)
                            if global_part is None:
                                has_global_parts = False
                            else:
                                global_parts.append(global_part)
                        wf_sims = torch.cat(wf_parts, dim=-1)
                        global_sims = torch.cat(global_parts, dim=-1) if has_global_parts else None
                        fused_sims = self._fuse_similarity_v3(wf_sims, global_sims)
                    else:
                        fused_sims = self._compute_similarity_v3(
                            text_wf_by_proto[0],
                            total_vid_embeds,
                            eval_vid_ids,
                            text_global_by_proto[0],
                        )

                    block_sizes = [block.shape[0] for block in vid_embeds_arr]
                    block_start = sum(block_sizes[:n_task])
                    block_end = block_start + block_sizes[n_task]
                    sims = fused_sims[..., block_start:block_end]

                    flat_text_arr = []
                    for arr in text_wf_by_proto:
                        flat_text_arr.extend(arr)
                    self._v2_print_validation_debug(
                        'final_validate_v3',
                        n_task + 1,
                        task_id,
                        flat_text_arr,
                        eval_vid_embeds,
                        sims,
                        eval_vid_ids,
                    )
                    if self._v2_debug_enabled():
                        print(
                            f"[V3-FINAL] eval_task={n_task + 1} total_video={total_vid_embeds.shape[0]} "
                            f"block_start={block_start} block_end={block_end} "
                            f"fused_sims={self._v2_stats(fused_sims)} task_sims={self._v2_stats(sims)}"
                        )
                    res = self.metrics(sims)
                    if n_task == task_id - 1:
                        list_val_acc_ii.append(res['R1'])
                        print()
                        print(f"Validation R@1 List: {list_val_acc_ii}")
                        print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                    elif n_task < task_id - 1:
                        if len(list_val_acc_ii) == 0:
                            print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                        else:
                            forgetting = list_val_acc_ii[n_task] - res['R1']
                            print()
                            print(f"Validation R@1 List: {list_val_acc_ii}")
                            print(f"Task {n_task + 1} R@1: {res['R1']:.6f}")
                            print(f"Task {n_task + 1} Forgetting: {forgetting:.6f}")
                            BWF.update(forgetting, num_classes)
                            print(f"Task {n_task + 1} BWF: {BWF.avg:.6f}")
        update_exp_result(self.task_log, task_id, bwf=BWF.avg)
        update_exp_result(self.overall_log, task_id, bwf=BWF.avg)
        if task_id == 1 and self.config.task_prototype:
            self._duplicate_weights()
            self._save_checkpoint(0, save_best=True)
        return BWF.avg

    def _load_checkpoint(self, model_name, task=None):
        """
        Load from saved checkpoints
        :param model_name: Model name experiment to be loaded
        """
        if task is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"Task{task}", model_name)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        state_dict = checkpoint['state_dict']
        
        self.model.load_state_dict(state_dict)
        print("Checkpoint loaded")

        if 'list_val_acc_ii' in checkpoint:
            self.list_val_acc_ii = checkpoint['list_val_acc_ii']

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, save checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            # 'config': self.config, 
            'task_id': self.task_id,
            'list_val_acc_ii': self.list_val_acc_ii,
        }

        save_path = self.checkpoint_dir

        if save_best:
            best_path = os.path.join(save_path, f'task{self.task_id}_model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
        else:
            save_dir = os.path.join(save_path, 'backup')
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f'checkpoint-task-{self.task_id}-epoch-{epoch}.pth')
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))

    def _duplicate_weights(self):
        """
        Duplicate weights across experts (Only for frame_fusion_moe)
        """
        for i in range(len(self.model.clipmodel.text_model.encoder.layers)):
            for proj in ['q', 'k', 'v', 'out']:
                # Get source weights from first expert
                source_weights = getattr(self.model.clipmodel.text_model.encoder.layers[i].self_attn, f"{proj}_lora").lora_Bs[0].weight.data
                # Copy weights to all other experts
                with torch.no_grad():
                    for j in range(1, len(getattr(self.model.clipmodel.text_model.encoder.layers[i].self_attn, f"{proj}_lora").lora_Bs)):
                        getattr(self.model.clipmodel.text_model.encoder.layers[i].self_attn, f"{proj}_lora").lora_Bs[j].weight.data.copy_(source_weights)

        print("Successfully duplicated weights from first expert to all other experts")
