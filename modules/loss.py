import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, List
from .tokenizer import clip_tokenizer

class CLIPPromptLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale, prompt_sim):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        prompt_log_sm = F.log_softmax(prompt_sim, dim=0)
        prompt_loss = -prompt_log_sm.mean()

        return (t2v_loss + v2t_loss) / 2.0 + 1.0 * prompt_loss

# X-Pool
class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0 

# Switch Prompt
class CaptionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        weight = torch.ones(clip_tokenizer.vocab_size)
        frequent_words = "in on of to about this that a an the and there here is are . , \
                          <|endoftext|> <|startoftext|>"
        frequent_ids = clip_tokenizer.convert_tokens_to_ids(
            clip_tokenizer.tokenize(frequent_words)
        )
        weight[frequent_ids] = config.frequent_word_weight
        self.register_buffer('weight', weight)
        self.mult = config.caption_loss_mult

    def forward(self, pred_logits, input_ids):
        mask = input_ids[:, :-1] != clip_tokenizer.eos_token_id
        pred_logits = pred_logits[mask]
        target_ids = input_ids[:, 1:][mask]
        return F.cross_entropy(pred_logits, 
                               target_ids, 
                               weight=self.weight) * self.mult

class NCELearnableTempLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(NCELearnableTempLoss, self).__init__()

    def forward(self, vis_feat, text_feat, temp):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        return loss

class TripletLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(TripletLoss, self).__init__()

    # def class_wise_contrastive_loss(self, vis_feat, text_feat, categories, temp, margin=0.5):

    #     # Convert categories to tensor of indices
    #     unique_categories = list(set(categories))
    #     category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    #     vis_labels = torch.tensor([category_to_idx[cat] for cat in categories], 
    #                             device=vis_feat.device)
    #     text_labels = vis_labels  # Since they share the same categories
        
    #     # Calculate similarity matrix
    #     logit_scale = temp.exp()
    #     sim_matrix = torch.matmul(vis_feat, text_feat.t()) * logit_scale
        
    #     # Create label matrix
    #     label_matrix = (vis_labels.unsqueeze(1) == text_labels.unsqueeze(0)).float()
        
    #     # Compute positive and negative losses
    #     pos_loss = (-sim_matrix * label_matrix).sum() / (label_matrix.sum() + 1e-6)
    #     neg_loss = (torch.clamp(sim_matrix - margin, min=0.0) * (1 - label_matrix)).sum() / ((1 - label_matrix).sum() + 1e-6)
        
    #     return pos_loss + neg_loss

    def forward(self, vis_feat, text_feat, temp, ref_vis_feat, scale=0.5, category=None):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        NCE_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()

        # Triplet loss
        if ref_vis_feat is not None:
            # Calculate the similarity between the text and the positive video
            txt_img_sim = torch.matmul(text_feat, vis_feat.permute(1, 0)) * logit_scale
            txt_neg_sim = torch.matmul(text_feat, ref_vis_feat.permute(1, 0)) * logit_scale
            labels = torch.arange(txt_img_sim.shape[0], device=txt_img_sim.device)

            # Calculate the triplet loss
            triplet_loss = F.cross_entropy(torch.cat([txt_img_sim, txt_neg_sim], dim=-1), labels)
            loss = (1-scale) * NCE_loss + scale * triplet_loss
        else:
            loss = NCE_loss

        return loss
    


def batch_word_frame_similarity_variable_text_v4(
    frames: torch.Tensor,                # [B, N, D]
    middle_embeds: List[torch.Tensor],   # len B, each [L_i, D]
    eps: float = 1e-8,
    temp: float = 0.07
) -> torch.Tensor:
    """
    Compute word–frame similarity for variable-length texts (soft alignment),
    supporting GPU execution and autograd.
    """
    device = frames.device
    num_videos, N, D = frames.shape
    num_texts = len(middle_embeds)

    sims_rows = []  # collect similarity rows for each video

    for i in range(num_videos):
        # Normalize to avoid magnitude explosion
        f_i = F.normalize(frames[i], dim=-1, eps=eps)  # [N, D]
        row_sims = []

        for j in range(num_texts):
            t_j = middle_embeds[j].to(device)
            if t_j.numel() == 0:
                row_sims.append(torch.tensor(0.0, device=device, dtype=frames.dtype))
                continue

            t_j = F.normalize(t_j, dim=-1, eps=eps)  # [L_j, D]
            S = torch.matmul(f_i, t_j.T)  # [N, L_j]

            # Prevent numerical overflow
            S_scaled = torch.clamp(S / 1.0, -50, 50)

            # soft alignment (smoothed alignment)
            alpha_w = torch.softmax(S_scaled, dim=0)  # frame weights for each word
            alpha_f = torch.softmax(S_scaled, dim=1)  # word weights for each frame

            # Compute smooth bidirectional similarity
            soft_over_frames = (alpha_w * S).sum(dim=0).mean()
            soft_over_words  = (alpha_f * S).sum(dim=1).mean()
            sim_ij = 0.5 * (soft_over_frames + soft_over_words)

            row_sims.append(sim_ij)

        row_sims = torch.stack(row_sims).to(device)
        sims_rows.append(row_sims)

    sims = torch.stack(sims_rows).to(device)
    return sims

# V5 is the rollback-safe training variant: it matches validator.py's
# bidirectional max-pooling similarity while keeping v4 untouched.
def batch_word_frame_similarity_variable_text_v5(
    frames: torch.Tensor,                # [B, N, D]
    middle_embeds: List[torch.Tensor],   # len B, each [L_i, D]
    eps: float = 1e-8,
    return_debug: bool = False
) -> torch.Tensor:
    """
    Compute the same bidirectional max-pooling word-frame similarity used by
    validation, but return a [num_videos, num_texts] matrix for training.
    """
    device = frames.device
    num_videos, _, _ = frames.shape
    num_texts = len(middle_embeds)

    # Debug option: debug collection is disabled by default and only
    # runs when SCL_and_CRP_v2 explicitly requests return_debug=True.
    debug = None
    if return_debug:
        debug = {
            'valid_frames': [], 'valid_words': [],
            'frame_norm_mean': [], 'word_norm_mean': [],
            'word_to_frame_term': [], 'frame_to_word_term': [],
            'pair_sim_mean': [], 'pair_sim_std': [],
        }
        debug['valid_frames'].append(float(frames.shape[1]))
        debug['frame_norm_mean'].append(frames.detach().float().norm(dim=-1).mean().item())

    frames = F.normalize(frames, dim=-1, eps=eps)
    sims_rows = []

    for video_idx in range(num_videos):
        frame_embed = frames[video_idx]  # [N, D]
        row_sims = []
        for text_idx in range(num_texts):
            text_embed = middle_embeds[text_idx].to(device)
            if text_embed.numel() == 0:
                row_sims.append(torch.tensor(0.0, device=device, dtype=frames.dtype))
                if return_debug:
                    debug['valid_words'].append(0.0)
                continue

            if return_debug:
                debug['valid_words'].append(float(text_embed.shape[0]))
                debug['word_norm_mean'].append(text_embed.detach().float().norm(dim=-1).mean().item())

            text_embed = F.normalize(text_embed, dim=-1, eps=eps)  # [L, D]
            similarity = torch.matmul(frame_embed, text_embed.T)  # [N, L]
            max_over_frames = similarity.max(dim=0).values.mean()
            max_over_words = similarity.max(dim=1).values.mean()
            if return_debug:
                sim_detached = similarity.detach().float()
                debug['word_to_frame_term'].append(max_over_frames.detach().float().item())
                debug['frame_to_word_term'].append(max_over_words.detach().float().item())
                debug['pair_sim_mean'].append(sim_detached.mean().item())
                debug['pair_sim_std'].append(sim_detached.std(unbiased=False).item())
            row_sims.append(0.5 * (max_over_frames + max_over_words))

        sims_rows.append(torch.stack(row_sims))

    sims = torch.stack(sims_rows)
    if return_debug:
        return sims, debug
    return sims


def KL_loss(student_logits, teacher_logits, T=2.0):
    # soft targets from teacher
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    # soft predictions from student
    student_log_probs = F.log_softmax(student_logits / T, dim=1)

    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)
    return loss


# StructAlign ETF refactor note: keep parameter/state ownership in the trainer/model,
# and move only the pure ETF loss computation here to preserve experiment behavior.
def compute_etf_alignment_loss_base(
    text_embeds_list,
    video_embeds,
    categories,
    etf,
    etf_proj1,
    etf_proj2,
):
    """Compute the base ETF alignment loss without owning any ETF parameters or state."""
    prototypes = etf[:, categories].T  # (B, D)

    text_reprs = []
    video_reprs = []

    for i in range(len(text_embeds_list)):
        w = text_embeds_list[i]
        f = video_embeds[i]
        p = prototypes[i]

        w_proj = etf_proj1(w)
        f_proj = etf_proj2(f)

        alpha = F.softmax(torch.matmul(w_proj, p), dim=0)
        beta = F.softmax(torch.matmul(f_proj, p), dim=0)

        w_bar = torch.sum(alpha.unsqueeze(-1) * w_proj, dim=0)
        f_bar = torch.sum(beta.unsqueeze(-1) * f_proj, dim=0)

        text_reprs.append(w_bar)
        video_reprs.append(f_bar)

    text_repr = torch.stack(text_reprs, dim=0)
    video_repr = torch.stack(video_reprs, dim=0)

    text_norm = F.normalize(text_repr, dim=-1)
    video_norm = F.normalize(video_repr, dim=-1)
    proto_norm = F.normalize(prototypes, dim=-1)

    text_loss = 1.0 - torch.sum(text_norm * proto_norm, dim=-1)
    video_loss = 1.0 - torch.sum(video_norm * proto_norm, dim=-1)

    return (text_loss + video_loss).mean()


def compute_etf_alignment_loss_incremental(
    text_embeds_list,
    video_embeds,
    categories,
    etf,
    etf_proj1,
    etf_proj2,
    text_proto_aug,
    video_proto_aug,
    proto_aug_label,
):
    """Compute the incremental ETF alignment loss without moving ETF ownership out of the trainer/model."""
    video_proto_aug = etf_proj1(video_proto_aug)
    text_proto_aug = etf_proj2(text_proto_aug)

    proto_target = etf[:, proto_aug_label].T
    proto_text_norm = F.normalize(text_proto_aug, dim=-1)
    proto_video_norm = F.normalize(video_proto_aug, dim=-1)
    proto_target_norm = F.normalize(proto_target, dim=-1)
    proto_text_loss = 1 - (proto_text_norm * proto_target_norm).sum(dim=-1)
    proto_video_loss = 1 - (proto_video_norm * proto_target_norm).sum(dim=-1)

    loss_real = compute_etf_alignment_loss_base(
        text_embeds_list=text_embeds_list,
        video_embeds=video_embeds,
        categories=categories,
        etf=etf,
        etf_proj1=etf_proj1,
        etf_proj2=etf_proj2,
    )
    loss_proto = (proto_text_loss + proto_video_loss).mean()

    return (loss_real + loss_proto).mean()



class SCL_and_CRP(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(SCL_and_CRP, self).__init__()

    
    def forward(self, vis_feat, text_feat, temp, ref_temp, ref_vis_feat,  ref_video_embeds, ref_text_embeds, scale=0.5, category=None):
        logit_scale = temp.exp()
        ref_logit_scale = ref_temp.exp()

        v2t = batch_word_frame_similarity_variable_text_v4(vis_feat, text_feat)
        t2v = v2t.permute(1, 0)
        D = t2v.size(-1)
        t2v = t2v / (D ** 0.5)
        v2t = v2t / (D ** 0.5)
        
        t2v = t2v * logit_scale
        v2t = v2t * logit_scale
        
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        NCE_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        distill_loss = None

        if (ref_video_embeds is not None) and (ref_text_embeds is not None):
           
            ref_v2t = batch_word_frame_similarity_variable_text_v4(ref_video_embeds, ref_text_embeds)
            ref_t2v = ref_v2t.permute(1, 0)
            
            ref_t2v = ref_t2v / (D ** 0.5)
            ref_v2t = ref_v2t / (D ** 0.5)
            
            ref_t2v = ref_t2v * ref_logit_scale
            ref_v2t = ref_v2t * ref_logit_scale

            distill_loss = (KL_loss(t2v, ref_t2v) + KL_loss(v2t, ref_v2t)) / 2.0
            loss = NCE_loss + distill_loss * 10.0  # 0 1  5  10 20  
        else:
            loss = NCE_loss

        if self.debug_enabled:
            self.last_debug_stats = {
                'loss_t2v': self._safe_item(NCE_loss),
                'loss_crp': self._safe_item(distill_loss),
                'loss_before_etf': self._safe_item(loss),
                'logit_scale': self._safe_item(logit_scale),
                'ref_logit_scale': self._safe_item(ref_logit_scale),
                't2v': self._sim_stats(t2v),
                'v2t': self._sim_stats(v2t),
                'wf_valid_words': self._list_stats(wf_debug.get('valid_words') if wf_debug else None),
                'wf_valid_frames': self._list_stats(wf_debug.get('valid_frames') if wf_debug else None),
                'wf_word_norm': self._list_stats(wf_debug.get('word_norm_mean') if wf_debug else None),
                'wf_frame_norm': self._list_stats(wf_debug.get('frame_norm_mean') if wf_debug else None),
                'wf_word_to_frame': self._list_stats(wf_debug.get('word_to_frame_term') if wf_debug else None),
                'wf_frame_to_word': self._list_stats(wf_debug.get('frame_to_word_term') if wf_debug else None),
                'wf_pair_sim_mean': self._list_stats(wf_debug.get('pair_sim_mean') if wf_debug else None),
                'wf_pair_sim_std': self._list_stats(wf_debug.get('pair_sim_std') if wf_debug else None),
            }
        return loss


# V2 keeps the original SCL_and_CRP loss intact so experiments can switch
# back by restoring the old config loss name.
class SCL_and_CRP_v2(nn.Module):
    """
    Contrastive loss variant that uses the same max-pooling word-frame
    similarity as validation for both current-task learning and distillation.
    """
    def __init__(self):
        super(SCL_and_CRP_v2, self).__init__()
        # Debug option: trainer toggles this flag for sparse diagnostics.
        self.debug_enabled = False
        self.last_debug_stats = {}

    def _safe_item(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().float().item()
        return float(value)

    def _list_stats(self, values):
        if values is None or len(values) == 0:
            return {}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {'mean': tensor.mean().item(), 'std': tensor.std(unbiased=False).item(), 'min': tensor.min().item(), 'max': tensor.max().item()}

    def _sim_stats(self, sims):
        stats = {}
        with torch.no_grad():
            val = sims.detach().float()
            stats.update({'shape': tuple(val.shape), 'min': val.min().item(), 'max': val.max().item(), 'mean': val.mean().item(), 'std': val.std(unbiased=False).item()})
            if val.dim() == 2 and val.shape[0] == val.shape[1]:
                diag = torch.diag(val)
                offdiag = val[~torch.eye(val.shape[0], dtype=torch.bool, device=val.device)]
                stats['diag_mean'] = diag.mean().item()
                stats['offdiag_mean'] = offdiag.mean().item() if offdiag.numel() > 0 else float('nan')
                stats['diag_minus_offdiag'] = stats['diag_mean'] - stats['offdiag_mean']
        return stats

    def forward(self, vis_feat, text_feat, temp, ref_temp, ref_vis_feat, ref_video_embeds, ref_text_embeds, scale=0.5, category=None):
        logit_scale = temp.exp()
        ref_logit_scale = ref_temp.exp()

        self.last_debug_stats = {}
        if self.debug_enabled:
            v2t, wf_debug = batch_word_frame_similarity_variable_text_v5(vis_feat, text_feat, return_debug=True)
        else:
            v2t = batch_word_frame_similarity_variable_text_v5(vis_feat, text_feat)
            wf_debug = None
        t2v = v2t.permute(1, 0)
        D = t2v.size(-1)
        t2v = t2v / (D ** 0.5)
        v2t = v2t / (D ** 0.5)

        t2v = t2v * logit_scale
        v2t = v2t * logit_scale

        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        NCE_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        distill_loss = None

        if (ref_video_embeds is not None) and (ref_text_embeds is not None):
            ref_v2t = batch_word_frame_similarity_variable_text_v5(ref_video_embeds, ref_text_embeds)
            ref_t2v = ref_v2t.permute(1, 0)

            ref_t2v = ref_t2v / (D ** 0.5)
            ref_v2t = ref_v2t / (D ** 0.5)

            ref_t2v = ref_t2v * ref_logit_scale
            ref_v2t = ref_v2t * ref_logit_scale

            distill_loss = (KL_loss(t2v, ref_t2v) + KL_loss(v2t, ref_v2t)) / 2.0
            loss = NCE_loss + distill_loss * 10.0
        else:
            loss = NCE_loss

        # Debug option: read-only diagnostics, enabled only by trainer.debug_v2.
        if self.debug_enabled:
            self.last_debug_stats = {
                'loss_t2v': self._safe_item(NCE_loss),
                'loss_crp': self._safe_item(distill_loss),
                'loss_before_etf': self._safe_item(loss),
                'logit_scale': self._safe_item(logit_scale),
                'ref_logit_scale': self._safe_item(ref_logit_scale),
                't2v': self._sim_stats(t2v),
                'v2t': self._sim_stats(v2t),
                'wf_valid_words': self._list_stats(wf_debug.get('valid_words') if wf_debug else None),
                'wf_valid_frames': self._list_stats(wf_debug.get('valid_frames') if wf_debug else None),
                'wf_word_norm': self._list_stats(wf_debug.get('word_norm_mean') if wf_debug else None),
                'wf_frame_norm': self._list_stats(wf_debug.get('frame_norm_mean') if wf_debug else None),
                'wf_word_to_frame': self._list_stats(wf_debug.get('word_to_frame_term') if wf_debug else None),
                'wf_frame_to_word': self._list_stats(wf_debug.get('frame_to_word_term') if wf_debug else None),
                'wf_pair_sim_mean': self._list_stats(wf_debug.get('pair_sim_mean') if wf_debug else None),
                'wf_pair_sim_std': self._list_stats(wf_debug.get('pair_sim_std') if wf_debug else None),
            }
        return loss


class NCELearnableTempLoss_lwf(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(NCELearnableTempLoss_lwf, self).__init__()

    def forward(self, vis_feat, text_feat, temp, ref_vis_feat, ref_text_feat):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        contrastive_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()

        # LwF loss
        curr_t2v = torch.matmul(vis_feat, ref_text_feat.permute(1, 0)) * logit_scale
        curr_v2t = curr_t2v.permute(1, 0)
        ref_t2v = torch.matmul(ref_vis_feat, ref_text_feat.permute(1, 0)) * logit_scale  # temperature
        ref_v2t = ref_t2v.permute(1, 0)
        distill_loss = (distillation(ref_t2v, curr_t2v) + distillation(ref_v2t, curr_v2t)).mean()

        loss = contrastive_loss + distill_loss

        return loss

def distillation(t, s, T=2.0):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
    return loss

class NCELearnableTempLoss_zscl(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(NCELearnableTempLoss_zscl, self).__init__()

    def forward(self, vis_feat, text_feat, temp, ref_vis_feat_curr, ref_vis_feat, ref_text_feat):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        contrastive_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()

        # ZSCL loss
        curr_t2v = torch.matmul(ref_vis_feat_curr, ref_text_feat.permute(1, 0)) * logit_scale
        curr_v2t = curr_t2v.permute(1, 0)
        ref_t2v = torch.matmul(ref_vis_feat, ref_text_feat.permute(1, 0)) * logit_scale  # temperature
        ref_v2t = ref_t2v.permute(1, 0)
        distill_loss = (distillation(ref_t2v, curr_t2v) + distillation(ref_v2t, curr_v2t)).mean()

        loss = contrastive_loss + distill_loss

        return loss

class NCELearnableTempLoss_moe(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(NCELearnableTempLoss_moe, self).__init__()

    def forward(self, vis_feat, text_feat, temp, router_weight, prev_router_weight, scale):

        # def print_grad(grad):
        #     print("Gradient flowing through router_weight:", grad)
        
        # router_weight.register_hook(print_grad)

        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        contrastive_loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        
        # Check MoE loss gradients
        moe_loss = torch.tensor(0.0, device=contrastive_loss.device, dtype=contrastive_loss.dtype)

        if len(prev_router_weight) == 0:
            return contrastive_loss, contrastive_loss, moe_loss
        else:
            num_tasks = len(prev_router_weight)
            
            for i, e_old in enumerate(prev_router_weight):
                # Compute similarity for this task
                similarity = (router_weight * e_old).sum(dim=-1).sum()
                moe_loss += similarity

            log_router_weight = F.log_softmax(router_weight, dim=-1)
            for i, e_old in enumerate(prev_router_weight):
                prev_prob = F.softmax(e_old, dim=-1)
                kl_div = F.kl_div(log_router_weight, prev_prob, reduction='sum')
                moe_loss -= kl_div

            moe_loss /= num_tasks
            
            # Final loss
            scale_tensor = torch.tensor(scale, device=moe_loss.device, dtype=moe_loss.dtype)
            total_loss = contrastive_loss + scale_tensor * moe_loss

            return total_loss, contrastive_loss, scale_tensor * moe_loss

class NCELearnableTempLoss_vt_ft(nn.Module):
    """
    Compute contrastive loss: video-(sub,cap)
    """

    def __init__(self):
        super(NCELearnableTempLoss_vt_ft, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, caption_feat, temp):
        logit_scale = temp.exp()
        # V-T
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        # F-C
        v2t_3 = torch.matmul(img_feat, caption_feat.permute(1, 0)) * logit_scale  # temperature
        t2v_3 = v2t_3.permute(1, 0)
        t2v_label_3 = torch.arange(t2v_3.shape[0], device=t2v_3.device)
        v2t_label_3 = t2v_label_3

        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label) + \
            F.cross_entropy(t2v_3, t2v_label_3) + F.cross_entropy(v2t_3, v2t_label_3)).mean()
        
        img_loss = (F.cross_entropy(t2v_3, t2v_label_3) + F.cross_entropy(v2t_3, v2t_label_3)).mean()

        return loss, img_loss

class CLIPLoss_vt_ft(nn.Module):
    """
    Compute contrastive loss: video-(sub,cap)
    """

    def __init__(self):
        super(CLIPLoss_vt_ft, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, caption_feat, temp):
        logit_scale = temp.exp()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        vis_feat = vis_feat / vis_feat.norm(dim=-1, keepdim=True)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        caption_feat = caption_feat / caption_feat.norm(dim=-1, keepdim=True)

        # V-T
        vis_feat_pooled = vis_feat.permute(1,2,0)
        text_feat = text_feat.unsqueeze(1)
        t2v = torch.bmm(text_feat, vis_feat_pooled).squeeze(1) * logit_scale  # temperature
        t2v_log_sm = F.log_softmax(t2v, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(t2v, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        # F-C
        v2t_3 = torch.mm(caption_feat, img_feat.t()) * logit_scale  # temperature
        v2t_log_sm_3 = F.log_softmax(v2t_3, dim=0)
        v2t_neg_ce_3 = torch.diag(v2t_log_sm_3)
        v2t_loss_3 = -v2t_neg_ce_3.mean()

        t2v_log_sm_3 = F.log_softmax(v2t_3, dim=1)
        t2v_neg_ce_3 = torch.diag(t2v_log_sm_3)
        t2v_loss_3 = -t2v_neg_ce_3.mean()

        loss = (t2v_loss + v2t_loss + t2v_loss_3 + v2t_loss_3) / 4.0
        img_loss = (t2v_loss_3 + v2t_loss_3) / 2.0
        return loss, img_loss

class LossFactory:
    @staticmethod
    def get_loss(config):
        if config.loss == 'clip':
            return CLIPLoss()
        elif config.loss == 'clip_vt_ft':
            return CLIPLoss_vt_ft()
        elif config.loss == 'clip_prompt':
            return CLIPPromptLoss()
        elif config.loss == 'NCELearnableTempLoss':
            return NCELearnableTempLoss()
        elif config.loss == 'lwf':
            return NCELearnableTempLoss_lwf()
        elif config.loss == 'zscl':
            return NCELearnableTempLoss_zscl()
        elif config.loss == 'triplet':
            return TripletLoss()
        elif config.loss == 'SCL_and_CRP':
            return SCL_and_CRP()
        # Rollback path: keep both loss names registered and switch from config.
        elif config.loss == 'SCL_and_CRP_v2':
            return SCL_and_CRP_v2()
        elif config.loss == 'NCELearnableTempLoss_moe':
            return NCELearnableTempLoss_moe()
        elif config.loss == 'NCELearnableTempLoss_vt_ft':
            return NCELearnableTempLoss_vt_ft()
        elif config.loss == 'clip+caption':
            return {'clip': CLIPLoss(),
                    'caption': CaptionLoss(config)}
        else:
            raise NotImplemented
