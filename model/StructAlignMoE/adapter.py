import torch
import torch.nn as nn
import math
from collections import Counter

import torch
import torch.nn as nn
from einops import rearrange


# class FrameCrossAttention(nn.Module):
#     """
#     Improved Frame Cross Attention:
#     - Only frame tokens (CLS tokens) participate in cross-frame attention
#     - Patch tokens remain local (intra-frame)
#     - Q = shifted frame tokens, K/V = current frame tokens
#     - Clean and readable implementation
#     """
#     def __init__(self, config, embed_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.config = config
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim

#         self.scale = self.head_dim ** -0.5

#         # Attention projections
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#         self.dropout = dropout

#     def shift_frames(self, frame_tokens, forward=True):
#         """
#         frame_tokens: [B, N, C]
#         """
#         if forward:
#             shifted = torch.roll(frame_tokens, shifts=1, dims=1)
#             shifted[:, 0] = frame_tokens[:, 0]  # 第一帧保持自身
#         else:
#             shifted = torch.roll(frame_tokens, shifts=-1, dims=1)
#             shifted[:, -1] = frame_tokens[:, -1]  # 最后一帧保持自身
#         return shifted

#     def forward(self, hidden_states, inputs_size, layer_id):
#         """
#         hidden_states: [B, N*(1+L), C]
#         inputs_size: (N, L)
#         """
#         B, T, C = hidden_states.size()
#         N, L = inputs_size

#         # reshape
#         x = hidden_states.view(B, N, 1+L, C)
#         frame_tokens = x[:, :, 0, :]      # [B, N, C]
#         patch_tokens = x[:, :, 1:, :]     # [B, N, L, C]

#         # decide direction
#         forward = (layer_id % 2 == 0)

#         # shifted frame tokens only
#         guide_tokens = self.shift_frames(frame_tokens, forward=forward)  # [B, N, C]

#         # project Q/K/V
#         Q = self.q_proj(guide_tokens)  # [B, N, C]
#         K = self.k_proj(frame_tokens)  # [B, N, C]
#         V = self.v_proj(frame_tokens)  # [B, N, C]

#         # multi-head reshape
#         Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads) * self.scale
#         K = rearrange(K, "b n (h d) -> b h n d", h=self.num_heads)
#         V = rearrange(V, "b n (h d) -> b h n d", h=self.num_heads)

#         # frame-level cross attention
#         attn = torch.matmul(Q, K.transpose(-1, -2))    # [B, H, N, N]
#         attn = attn.softmax(dim=-1)
#         attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
#         frame_out = torch.matmul(attn, V)              # [B, H, N, D]
#         frame_out = rearrange(frame_out, "b h n d -> b n (h d)")

#         # replace frame tokens + keep patch tokens
#         out = torch.cat([frame_out.unsqueeze(2), patch_tokens], dim=2)   # [B, N, 1+L, C]
#         out = out.reshape(B, N*(1+L), C)

#         return self.out_proj(out)


class FrameCrossAttention(nn.Module):
    def __init__(self, config, embed_dim, num_heads, dropout):
        super(FrameCrossAttention, self).__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, inputs_size, layer_id):
        """
        hidden_states: [B, N*(1+L), C]
        inputs_size: (N, L)
        layer_id: int
        """
        N, L = inputs_size  # Number of frames and patches
        bsz, tgt_len, embed_dim = hidden_states.size()

        # Handle the guided feature
        guide_states = hidden_states.view(bsz, N, 1+L, embed_dim)
        
        if layer_id % 2 == 0:
            # Forward processing: use previous frame's hidden states as guide
            guide_states = torch.roll(guide_states, shifts=1, dims=1)
            guide_states[:, 0] = hidden_states.view(bsz, N, 1+L, embed_dim)[:, 0]  # First frame keeps original
        else:
            # Backward processing: use next frame's hidden states as guide
            guide_states = torch.roll(guide_states, shifts=-1, dims=1)
            guide_states[:, -1] = hidden_states.view(bsz, N, 1+L, embed_dim)[:, -1]  # Last frame keeps original

        # Project guide states to query space after shifting
        guide_states = guide_states.view(bsz, N*(1+L), embed_dim)

        # Linear projection
        query_states = self.q_proj(guide_states) * self.scale
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._shape(query_states, tgt_len, bsz) # [bsz, num_heads, N*(1+L), head_dim]
        key_states = self._shape(key_states, tgt_len, bsz)  
        value_states = self._shape(value_states, tgt_len, bsz)

        # Reshape for attention computation [B*num_heads*N, 1+L, head_dim]
        query_states = query_states.view(bsz * self.num_heads * N, 1+L, self.head_dim)  
        key_states = key_states.view(bsz * self.num_heads * N, 1+L, self.head_dim)     
        value_states = value_states.view(bsz * self.num_heads * N, 1+L, self.head_dim)  

        # Compute inter-frame attention attention
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # [B*num_heads*N, 1+L, 1+L]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute attention output
        attn_output = torch.bmm(attn_probs, value_states)

        # Reshape back to original dimensions [bsz, N*(1+L), embed_dim]
        attn_output = attn_output.view(bsz, self.num_heads, N, 1+L, self.head_dim)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(bsz, N*(1+L), embed_dim)

        return self.out_proj(attn_output)


class LoRAAdapter(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int, 
                 r: int = 8,
                 lora_alpha: int = 16,
                 lora_nums: int = 2,
                 topk: int = 2,
                 lora_dropout: float = 0.1):
        super().__init__()
        
        self.r = r
        self.lora_nums = lora_nums
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        if self.lora_nums < topk:
            self.topk = self.lora_nums
        else:
            self.topk = topk
            
        self.choose_map = torch.zeros([self.lora_nums]) # count the number of samples that each expert gets

        self.dropout = nn.Dropout(p=lora_dropout)
        self.softplus = nn.Softplus()
        
        self.noisy_gating = True
        self.is_train = True

        # LoRA matrices - one A and multiple Bs
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_Bs = nn.ModuleList([
            nn.Linear(r, out_features, bias=False) 
            for _ in range(lora_nums)
        ])
        
        # Routing components
        self.lora_route = nn.Linear(in_features, lora_nums, bias=False)
        self.w_noise = nn.Linear(in_features, lora_nums, bias=False)
        
        # Initialize weights
        self.reset_parameters()
        # self.register_gradient_hooks()

    def reset_choose_map(self):
        self.choose_map.zero_()  

    def reset_parameters(self):
        # Initialize router and noise
        nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_noise.weight)
        
        # Initialize LoRA matrices 
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        for lora_B in self.lora_Bs:
            nn.init.zeros_(lora_B.weight)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.is_train = True
        else:
            self.is_train = False

    def eval(self):
        return self.train(False)
    
    def noisy_top_k_gating(self, x, train=True, noise_epsilon=1e-2):
        clean_logits = self.lora_route(x)
        
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)  
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            
        # Get top-k gates
        top_logits, top_indices = logits.topk(min(self.topk + 1, self.lora_nums), dim=-1)
        top_k_logits = top_logits[:, :self.topk] 
        top_k_indices = top_indices[:, :self.topk]
        top_k_gates = nn.functional.softmax(top_k_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        
        # Expand gates to full size
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        return gates

    def register_gradient_hooks(self):
        self.gradient_stats = {}
        
        def hook_fn(name):
            def fn(grad):
                if grad is not None:
                    self.gradient_stats[name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'zero_frac': (grad == 0).float().mean().item()
                    }
                return grad
            return fn
        
        # Register hooks for key parameters
        self.lora_route.weight.register_hook(hook_fn('route'))
        # self.lora_A.weight.register_hook(hook_fn('lora_A'))
        # for i, B in enumerate(self.lora_Bs):
        #     B.weight.register_hook(hook_fn(f'lora_B_{i}'))

    # def forward(self, x: torch.Tensor, eof_index = None, task_prototype = None, task_id = None):
    #     if task_id ==1 and task_prototype is None:
    #         lora_output = self.lora_Bs[0](self.lora_A(self.dropout(x)))
    #     else:       
    #         # MoE routing input
    #         if eof_index is not None:
    #             routing_input = x[torch.arange(x.size(0)), eof_index]
    #         else:
    #             routing_input = x
                
    #         # Get gating weights
    #         gates = self.noisy_top_k_gating(routing_input, self.is_train)

    #         # Step 2: count the frequency of each expert being selected
    #         if not self.is_train:
    #             nonzero_indices = torch.nonzero(gates) # index of selected experts
    #             counter = Counter(nonzero_indices[:, 1].tolist()) # count the number of samples that each expert gets
    #             for number, count in counter.items():
    #                 self.choose_map[number] = self.choose_map[number] + count

    #         # Apply LoRA with gating
    #         shared_output = self.lora_A(self.dropout(x))
            
    #         outputs = []
    #         for i in range(self.lora_nums):
    #             if gates[:, i].any(): 
    #                 expert_output = self.lora_Bs[i](shared_output)
    #                 outputs.append(expert_output * gates[:, i].view(-1, 1, 1))

    #         lora_output = sum(outputs)
                
    #     return lora_output * self.scaling
    
    def forward(self, x: torch.Tensor, eof_index = None, task_prototype = None, task_id = None):
        if task_id ==1 and task_prototype is not None:
            lora_output = self.lora_Bs[0](self.lora_A(self.dropout(x)))
        else:       
            # MoE routing input
            if eof_index is not None:
                routing_input = x[torch.arange(x.size(0)), eof_index]
                if task_prototype is not None:
                    routing_input = routing_input + task_prototype
            else:
                routing_input = x
                
            # Get gating weights
            gates = self.noisy_top_k_gating(routing_input, self.is_train)

            # Step 2: count the frequency of each expert being selected
            if not self.is_train:
                nonzero_indices = torch.nonzero(gates) # index of selected experts
                counter = Counter(nonzero_indices[:, 1].tolist()) # count the number of samples that each expert gets
                for number, count in counter.items():
                    self.choose_map[number] = self.choose_map[number] + count

            # Apply LoRA with gating
            shared_output = self.lora_A(self.dropout(x))
            
            outputs = []
            for i in range(self.lora_nums):
                if gates[:, i].any(): 
                    expert_output = self.lora_Bs[i](shared_output)
                    outputs.append(expert_output * gates[:, i].view(-1, 1, 1))

            lora_output = sum(outputs)
                
        return lora_output * self.scaling
    
