#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask2Former-based fusion model for camera-lidar segmentation.

Pure-PyTorch implementation — no Detectron2 dependency.

Architecture faithful to Cheng et al., "Masked-attention Mask Transformer for
Universal Image Segmentation", CVPR 2022:

  1. Shared timm backbone extracts multi-scale features.
  2. MSDeformAttnPixelDecoder: projects each scale to d_model, adds learnable
     level embeddings, then runs 6 transformer encoder layers of multi-scale
     deformable self-attention (F.grid_sample — no custom CUDA kernels).
     The finest scale is additionally up-convolved to produce mask_features.
  3. Mask2FormerDecoder: L=9 layers of masked cross-attention (each query
     attends only within its predicted foreground region from the previous
     layer) + self-attention + FFN, cycling multi-scale encoder memory.
     Prediction head uses a 3-layer MLP for mask embeddings (§A.2).
     Intermediate predictions enable deep supervision.
  4. Loss uses Hungarian matching on the *final* layer only; the same matched
     indices are reused for all auxiliary layers (official strategy).
  5. Dense segmentation:
       segmap[c] = Σ_q  softmax(class_logits_q)[c] · sigmoid(mask_q)

Camera-lidar fusion: element-wise addition of backbone features per scale via
ResidualConvUnit pairs (one per stream), matching the other fusion models.

Remaining intentional differences from the official codebase:
  - No panoptic/instance head — semantic segmentation only.
  - Camera-lidar dual-stream backbone (not in the original).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class ResidualConvUnit(nn.Module):
    """3×3 residual conv block."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


def _pos_encoding_2d(h: int, w: int, d: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal 2-D positional encoding → [1, H*W, d]."""
    assert d % 4 == 0, "d_model must be divisible by 4 for 2-D pos encoding"
    half = d // 2
    dim_t = torch.arange(half, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half)

    y = torch.arange(h, dtype=torch.float32, device=device)[:, None] / dim_t  # [H, d/2]
    x = torch.arange(w, dtype=torch.float32, device=device)[:, None] / dim_t  # [W, d/2]

    y[:, 0::2] = torch.sin(y[:, 0::2])
    y[:, 1::2] = torch.cos(y[:, 1::2])
    x[:, 0::2] = torch.sin(x[:, 0::2])
    x[:, 1::2] = torch.cos(x[:, 1::2])

    # Interleave y repeated for each column and x repeated for each row
    y_enc = y.unsqueeze(1).expand(h, w, half)   # [H, W, d/2]
    x_enc = x.unsqueeze(0).expand(h, w, half)   # [H, W, d/2]
    pos = torch.cat([y_enc, x_enc], dim=-1)      # [H, W, d]
    return pos.view(1, h * w, d)                  # [1, H*W, d]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Deformable Attention  (Zhu et al., ICLR 2021)
# Pure-PyTorch: uses F.grid_sample — no custom CUDA kernels required.
# ─────────────────────────────────────────────────────────────────────────────

class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Self/Cross Attention.

    For each query the network predicts:
      - n_heads × n_levels × n_points  2-D sampling offsets (pixel units)
      - n_heads × n_levels × n_points  scalar attention weights (softmax'd)

    Each (head, level, point) sampled feature is computed via bilinear
    interpolation (F.grid_sample), then weighted and summed.
    """

    def __init__(self, d_model: int = 256, n_levels: int = 3,
                 n_heads: int = 8, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_levels = n_levels
        self.n_heads  = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets  = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj        = nn.Linear(d_model, d_model)
        self.output_proj       = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid   = torch.stack([thetas.cos(), thetas.sin()], dim=-1)          # [H, 2]
        grid  /= grid.abs().max(dim=-1, keepdim=True).values
        grid   = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid.view(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self,
                query:             torch.Tensor,   # [B, Lq, d_model]
                reference_points:  torch.Tensor,   # [B, Lq, n_levels, 2] in [0,1] (y,x)
                value_flat:        torch.Tensor,   # [B, sum_HW, d_model]
                spatial_shapes:    torch.Tensor,   # [n_levels, 2] long (H_l, W_l)
                level_start_index: torch.Tensor,   # [n_levels] long
                ) -> torch.Tensor:                 # [B, Lq, d_model]
        B, Lq, _ = query.shape

        value = self.value_proj(value_flat)                           # [B, Lv, d]
        value = value.view(B, -1, self.n_heads, self.head_dim)        # [B, Lv, H, hd]

        offsets = self.sampling_offsets(query)                        # [B,Lq,H*L*P*2]
        offsets = offsets.view(B, Lq, self.n_heads, self.n_levels, self.n_points, 2)

        attn_w = self.attention_weights(query)                        # [B,Lq,H*L*P]
        attn_w = attn_w.view(B, Lq, self.n_heads, self.n_levels * self.n_points)
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = attn_w.view(B, Lq, self.n_heads, self.n_levels, self.n_points)

        # Normalize offsets (pixel units) to [0,1] relative to each level size
        shapes   = spatial_shapes.float().to(query.device)            # [L, 2]
        off_norm = shapes[None, None, None, :, None, :]               # [1,1,1,L,1,2]
        norm_off = offsets / off_norm                                  # [B,Lq,H,L,P,2]

        # reference_points [B,Lq,L,2] → [B,Lq,1,L,1,2]
        ref_exp       = reference_points.unsqueeze(2).unsqueeze(4)
        sampling_locs = ref_exp + norm_off                             # [B,Lq,H,L,P,2]

        # (y,x) in [0,1] → (x,y) in [-1,1] for F.grid_sample
        sampling_locs = sampling_locs[..., [1, 0]]
        sampling_locs = 2.0 * sampling_locs - 1.0

        output = torch.zeros(B, Lq, self.n_heads, self.head_dim,
                             device=query.device, dtype=query.dtype)

        for l_idx in range(self.n_levels):
            H_l = int(spatial_shapes[l_idx, 0])
            W_l = int(spatial_shapes[l_idx, 1])
            s   = int(level_start_index[l_idx])

            val_l = value[:, s:s + H_l * W_l]                        # [B, HW, H, hd]
            val_l = val_l.view(B, H_l, W_l, self.n_heads, self.head_dim)
            val_l = val_l.permute(0, 3, 4, 1, 2)                     # [B, H, hd, H_l, W_l]
            val_l = val_l.reshape(B * self.n_heads, self.head_dim, H_l, W_l)

            sl = sampling_locs[:, :, :, l_idx, :, :]                 # [B,Lq,H,P,2]
            sl = sl.permute(0, 2, 1, 3, 4)                           # [B,H,Lq,P,2]
            sl = sl.reshape(B * self.n_heads, 1, Lq * self.n_points, 2)

            sampled = F.grid_sample(val_l, sl, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
            sampled = sampled.squeeze(2)                              # [B*H, hd, Lq*P]
            sampled = sampled.view(B, self.n_heads, self.head_dim, Lq, self.n_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)                 # [B,Lq,H,P,hd]

            w = attn_w[:, :, :, l_idx, :].unsqueeze(-1)              # [B,Lq,H,P,1]
            output = output + (sampled * w).sum(dim=3)                # [B,Lq,H,hd]

        output = output.reshape(B, Lq, self.d_model)
        return self.output_proj(output)


class MSDeformAttnEncoderLayer(nn.Module):
    """Deformable self-attention encoder layer used inside the pixel decoder."""

    def __init__(self, d_model: int = 256, n_levels: int = 3,
                 n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.deform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, reference_points, spatial_shapes, level_start_index):
        attn_out = self.deform_attn(src, reference_points, src,
                                    spatial_shapes, level_start_index)
        src = self.norm1(src + attn_out)
        src = self.norm2(src + self.ffn(src))
        return src


class MSDeformAttnPixelDecoder(nn.Module):
    """
    Official Mask2Former pixel decoder.

    Steps:
      1. Project each backbone scale to out_channels (1×1 conv + GroupNorm).
      2. Add learnable level embeddings.
      3. Flatten + concatenate all scales → run n_encoder_layers of
         multi-scale deformable self-attention (queries = pixels, reference
         points = each pixel's normalised centre, broadcast across all levels).
      4. Finest scale → mask_features (high-res pixel embedding).
      5. Coarser scales → multi-scale memory for the transformer decoder.
    """

    def __init__(self, in_channels_list: list[int],
                 out_channels:     int = 256,
                 n_encoder_layers: int = 6,
                 n_heads:          int = 8,
                 n_points:         int = 4):
        super().__init__()
        n_levels = len(in_channels_list)

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1),
                nn.GroupNorm(min(32, out_channels), out_channels),
            )
            for c in in_channels_list
        ])

        self.level_embed = nn.Parameter(torch.zeros(n_levels, out_channels))
        nn.init.normal_(self.level_embed)

        self.encoder = nn.ModuleList([
            MSDeformAttnEncoderLayer(out_channels, n_levels=n_levels,
                                     n_heads=n_heads, n_points=n_points)
            for _ in range(n_encoder_layers)
        ])

        self.mask_features = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]):
        """
        Args:
            features : list of [B, C_i, H_i, W_i], fine→coarse.
        Returns:
            multi_scale : list of encoded coarser scale tensors.
            pixel_features : finest [B, out_channels, H_0, W_0].
        """
        B = features[0].shape[0]

        proj = [
            self.input_proj[i](f) + self.level_embed[i].view(1, -1, 1, 1)
            for i, f in enumerate(features)
        ]

        spatial_shapes = torch.tensor(
            [[f.shape[2], f.shape[3]] for f in proj],
            dtype=torch.long, device=proj[0].device
        )
        level_start_index = torch.zeros(len(proj), dtype=torch.long, device=proj[0].device)
        cumsum = 0
        for i, f in enumerate(proj):
            level_start_index[i] = cumsum
            cumsum += f.shape[2] * f.shape[3]

        src = torch.cat([f.flatten(2).permute(0, 2, 1) for f in proj], dim=1)

        # Reference points: normalised centre of each pixel, broadcast to all levels
        ref_list = []
        for H_l, W_l in spatial_shapes.tolist():
            ref_y = (torch.arange(H_l, dtype=torch.float32, device=src.device) + 0.5) / H_l
            ref_x = (torch.arange(W_l, dtype=torch.float32, device=src.device) + 0.5) / W_l
            gy, gx = torch.meshgrid(ref_y, ref_x, indexing='ij')
            ref_list.append(torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1))
        ref_flat = torch.cat(ref_list, dim=0)                          # [sum_HW, 2]
        n_levels  = len(proj)
        ref_pts   = ref_flat[:, None, :].expand(-1, n_levels, -1)      # [sum_HW, L, 2]
        ref_pts   = ref_pts.unsqueeze(0).expand(B, -1, -1, -1)         # [B, sum_HW, L, 2]

        for layer in self.encoder:
            src = layer(src, ref_pts, spatial_shapes, level_start_index)

        encoded = []
        for i, (H_l, W_l) in enumerate(spatial_shapes.tolist()):
            s = int(level_start_index[i])
            feat = src[:, s:s + H_l * W_l].permute(0, 2, 1).view(B, -1, H_l, W_l)
            encoded.append(feat)

        pixel_features = self.mask_features(encoded[0])
        multi_scale    = encoded[1:] if len(encoded) > 1 else encoded

        return multi_scale, pixel_features


# ─────────────────────────────────────────────────────────────────────────────
# Masked Attention Decoder
# ─────────────────────────────────────────────────────────────────────────────

class MaskedAttentionDecoderLayer(nn.Module):
    """Single Mask2Former decoder layer.

    Sequence (§3.2 of Cheng et al.):
      1. Masked cross-attention  (queries → selected memory pixels)
      2. Self-attention          (queries ↔ queries)
      3. Feed-forward network
    """

    def __init__(self, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.nhead   = nhead

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead,
                                                dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead,
                                               dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.0)

    def forward(self,
                queries:         torch.Tensor,   # [B, Q, d]
                memory:          torch.Tensor,   # [B, HW, d]
                attn_mask:       torch.Tensor,   # [B*nhead, Q, HW]  or [B, Q, HW]
                query_pos:       torch.Tensor,   # [B, Q, d] – sinusoidal query pos
                memory_pos:      torch.Tensor,   # [1, HW, d] – sinusoidal pixel pos
                ) -> torch.Tensor:

        # Expand attn_mask dim if needed [B, Q, HW] → [B*nhead, Q, HW]
        if attn_mask.dim() == 3 and attn_mask.shape[0] != queries.shape[0] * self.nhead:
            attn_mask = attn_mask.repeat_interleave(self.nhead, dim=0)  # [B*nhead, Q, HW]

        # 1. Masked cross-attention (pre-norm)
        q_ca = queries + query_pos
        kv_ca = memory + memory_pos.expand(memory.shape[0], -1, -1)
        ca_out, _ = self.cross_attn(
            query=q_ca, key=kv_ca, value=memory,
            attn_mask=attn_mask.to(queries.dtype),
        )
        queries = self.norm1(queries + self.dropout(ca_out))

        # 2. Self-attention (pre-norm)
        q_sa = queries + query_pos
        sa_out, _ = self.self_attn(query=q_sa, key=q_sa, value=queries)
        queries = self.norm2(queries + self.dropout(sa_out))

        # 3. FFN (pre-norm)
        queries = self.norm3(queries + self.dropout(self.ffn(queries)))

        return queries


class Mask2FormerDecoder(nn.Module):
    """Masked-Attention Transformer Decoder for Mask2Former.

    Stacks L decoder layers, cycling through multi-scale memory features
    (coarsest → finest, cycling) as in the paper.  After each layer a
    lightweight prediction head produces intermediate class logits + masks
    enabling deep supervision.
    """

    def __init__(self,
                 d_model:       int = 256,
                 nhead:         int = 8,
                 num_layers:    int = 9,
                 num_queries:   int = 100,
                 num_classes:   int = 4,
                 num_scales:    int = 3):
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers
        self.num_scales = num_scales

        # Learnable query embeddings: position embeddings (query_pos) and
        # content embeddings (queries themselves start at zero per §3.2).
        self.query_pos_embed = nn.Embedding(num_queries, d_model)  # fixed positional
        self.query_feat      = nn.Embedding(num_queries, d_model)  # learned content

        self.layers = nn.ModuleList([
            MaskedAttentionDecoderLayer(d_model=d_model, nhead=nhead)
            for _ in range(num_layers)
        ])

        # Prediction heads — shared weights across layers (weight-sharing in §A.2)
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        # 3-layer mask embedding MLP (paper §A.2 specifies 3 layers)
        self.mask_embed  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def _predict(self, queries: torch.Tensor,
                 pixel_features: torch.Tensor):
        """Predict class logits and binary masks from query features.

        Args:
            queries       : [B, Q, d]
            pixel_features: [B, d, H, W]
        Returns:
            class_logits : [B, Q, C+1]
            masks        : [B, Q, H, W]  (raw logits)
        """
        class_logits = self.class_embed(queries)              # [B, Q, C+1]
        mask_emb     = self.mask_embed(queries)               # [B, Q, d]
        masks = torch.einsum('bqd,bdhw->bqhw', mask_emb,
                             pixel_features) / math.sqrt(self.d_model)
        return class_logits, masks

    def forward(self,
                multi_scale_features: list[torch.Tensor],   # coarser FPN levels
                pixel_features:       torch.Tensor,         # finest [B, d, H, W]
                ):
        """
        Returns:
            all_class_logits : list[Tensor[B, Q, C+1]]  one per layer + init
            all_masks        : list[Tensor[B, Q, H, W]] one per layer + init
        """
        b = pixel_features.shape[0]
        device = pixel_features.device

        # Initialise queries
        queries   = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1)  # [B, Q, d]
        query_pos = self.query_pos_embed.weight.unsqueeze(0).expand(b, -1, -1)

        # Initial prediction from untrained queries (before any layer)
        cls0, masks0 = self._predict(queries, pixel_features)
        all_class_logits = [cls0]
        all_masks        = [masks0]

        num_scales = max(1, len(multi_scale_features))

        for layer_idx, layer in enumerate(self.layers):
            # Cycle through multi-scale memory (coarsest first, cycling)
            scale_idx = layer_idx % num_scales
            mem_feat  = multi_scale_features[scale_idx]  # [B, d, h_s, w_s]
            b_m, d_m, h_s, w_s = mem_feat.shape

            # Flatten memory for attention
            memory     = mem_feat.flatten(2).permute(0, 2, 1)  # [B, h_s*w_s, d]
            memory_pos = _pos_encoding_2d(h_s, w_s, d_m, device)   # [1, HW, d]

            # Build attention mask from previous prediction (binarise at threshold 0)
            prev_masks = all_masks[-1]                        # [B, Q, H_pf, W_pf]
            # Down-sample to current memory scale
            attn_map = F.interpolate(
                prev_masks.float(), size=(h_s, w_s), mode='bilinear', align_corners=False
            )  # [B, Q, h_s, w_s]
            # Boolean mask: True (attend) where sigmoid(prev) > 0.5
            # Per Mask2Former §3.2: mask is 0 for attended pixels, -inf otherwise
            attend = (attn_map.sigmoid() > 0.5).flatten(2)   # [B, Q, h_s*w_s]
            # If a query has no predicted foreground, lift the mask (attend everywhere)
            no_fg = ~attend.any(dim=-1, keepdim=True)         # [B, Q, 1]
            attend = attend | no_fg

            # Convert bool → float attention bias [B, Q, HW]
            float_mask = torch.zeros_like(attend, dtype=queries.dtype)
            float_mask[~attend] = float('-inf')

            # Run decoder layer
            queries = layer(
                queries=queries,
                memory=memory,
                attn_mask=float_mask,
                query_pos=query_pos,
                memory_pos=memory_pos,
            )

            cls_l, masks_l = self._predict(queries, pixel_features)
            all_class_logits.append(cls_l)
            all_masks.append(masks_l)

        return all_class_logits, all_masks


# ─────────────────────────────────────────────────────────────────────────────
# Loss criterion (identical bipartite matching, supports aux losses)
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerCriterion(nn.Module):
    """Bipartite matching loss for Mask2Former with deep supervision.

    Computes the Hungarian matching loss on the final layer predictions plus
    auxiliary losses on every intermediate layer, each weighted by
    ``aux_weight``.

    Per-batch loss = final_loss + aux_weight * mean(intermediate_losses)
    """

    def __init__(self,
                 num_classes:    int,
                 cost_class:     float = 2.0,
                 cost_mask:      float = 5.0,
                 cost_dice:      float = 5.0,
                 weight_class:   float = 2.0,
                 weight_mask:    float = 5.0,
                 weight_dice:    float = 5.0,
                 no_object_coef: float = 0.1,
                 aux_weight:     float = 0.5,
                 class_weights:  torch.Tensor | None = None):
        super().__init__()
        self.num_classes    = num_classes
        self.cost_class     = cost_class
        self.cost_mask      = cost_mask
        self.cost_dice      = cost_dice
        self.weight_class   = weight_class
        self.weight_mask    = weight_mask
        self.weight_dice    = weight_dice
        self.no_object_coef = no_object_coef
        self.aux_weight     = aux_weight

        if class_weights is not None:
            eos = torch.cat([class_weights,
                             class_weights.new_tensor([no_object_coef])])
        else:
            eos = None

        if eos is not None:
            self.register_buffer('ce_weight', eos)
        else:
            self.ce_weight = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _build_gt(self, anno_b: torch.Tensor):
        segments = []
        for c in range(self.num_classes):
            mask = (anno_b == c).float()
            if mask.sum() > 0:
                segments.append((c, mask))
        return segments

    @torch.no_grad()
    def _cost_matrix(self,
                     cls_logits_q: torch.Tensor,  # [Q, C+1]
                     pred_masks_q: torch.Tensor,  # [Q, h, w]
                     gt_classes:   list[int],
                     gt_masks:     torch.Tensor,  # [M, h, w]
                     ) -> torch.Tensor:            # [Q, M]
        Q = cls_logits_q.shape[0]
        M = len(gt_classes)
        dev = cls_logits_q.device

        cls_probs  = F.softmax(cls_logits_q, dim=-1)
        gt_idx     = torch.tensor(gt_classes, device=dev)
        class_cost = -cls_probs[:, gt_idx]

        pred_flat = pred_masks_q.flatten(1)
        pred_sig  = pred_flat.sigmoid()
        gt_flat   = gt_masks.flatten(1)
        HW        = pred_flat.shape[1]

        p_exp = pred_flat.unsqueeze(1).expand(Q, M, HW)
        g_exp = gt_flat.unsqueeze(0).expand(Q, M, HW)
        bce_cost = F.binary_cross_entropy_with_logits(
            p_exp, g_exp, reduction='none').mean(-1)

        num      = 2 * torch.einsum('qn,mn->qm', pred_sig, gt_flat)
        den      = pred_sig.sum(-1, keepdim=True) + gt_flat.sum(-1).unsqueeze(0) + 1e-5
        dice_cost = 1.0 - num / den

        return (self.cost_class * class_cost
                + self.cost_mask  * bce_cost
                + self.cost_dice  * dice_cost)

    def _layer_loss_with_indices(self,
                                  class_logits: torch.Tensor,   # [B, Q, C+1]
                                  pred_masks:   torch.Tensor,   # [B, Q, h, w]
                                  anno:         torch.Tensor,   # [B, H, W]
                                  indices:      list,           # per image: (row, col, gt_cls, gt_mask)
                                  ) -> torch.Tensor:
        """Compute loss for one decoder layer given pre-computed matching indices."""
        B, Q, _ = class_logits.shape
        _, _, h, w = pred_masks.shape
        dev = class_logits.device
        no_obj_idx = torch.tensor([self.num_classes], dtype=torch.long, device=dev)
        total_loss = class_logits.new_tensor(0.0)

        for b in range(B):
            row_ind, col_ind, gt_classes, gt_masks_ref = indices[b]

            if gt_masks_ref is None or len(row_ind) == 0:
                targets = no_obj_idx.expand(Q)
                total_loss = total_loss + self.no_object_coef * F.cross_entropy(
                    class_logits[b], targets, weight=self.ce_weight)
                continue

            # Rescale reference GT masks to this layer's resolution if needed
            if gt_masks_ref.shape[-1] != w or gt_masks_ref.shape[-2] != h:
                gt_masks = F.interpolate(
                    gt_masks_ref.unsqueeze(1).float(), size=(h, w), mode='nearest'
                ).squeeze(1).to(dev)
            else:
                gt_masks = gt_masks_ref.to(dev)

            matched    = set(int(r) for r in row_ind)
            batch_loss = class_logits.new_tensor(0.0)

            for r, c in zip(row_ind, col_ind):
                tgt = torch.tensor([gt_classes[c]], dtype=torch.long, device=dev)
                batch_loss = batch_loss + self.weight_class * F.cross_entropy(
                    class_logits[b, r].unsqueeze(0), tgt, weight=self.ce_weight)
                batch_loss = batch_loss + self.weight_mask * F.binary_cross_entropy_with_logits(
                    pred_masks[b, r], gt_masks[c])
                p_sig = pred_masks[b, r].sigmoid().flatten()
                g     = gt_masks[c].flatten()
                dice  = 1.0 - (2*(p_sig*g).sum() + 1e-5) / (p_sig.sum() + g.sum() + 1e-5)
                batch_loss = batch_loss + self.weight_dice * dice

            unmatched = torch.tensor([q for q in range(Q) if q not in matched],
                                     dtype=torch.long, device=dev)
            if unmatched.numel() > 0:
                batch_loss = batch_loss + self.no_object_coef * F.cross_entropy(
                    class_logits[b][unmatched], no_obj_idx.expand(unmatched.numel()),
                    weight=self.ce_weight)

            total_loss = total_loss + batch_loss

        return total_loss / B

    def forward(self,
                all_class_logits: list[torch.Tensor],
                all_masks:        list[torch.Tensor],
                anno:             torch.Tensor,
                ) -> torch.Tensor:
        """
        1. Run Hungarian matching on the *final* layer predictions only.
        2. Compute final layer loss from those matches.
        3. Reuse the same matched indices for every auxiliary layer loss
           (no re-matching per layer — matches the official implementation).
        """
        final_cls, final_masks = all_class_logits[-1], all_masks[-1]
        B, Q, _    = final_cls.shape
        _, _, h, w = final_masks.shape
        dev        = final_cls.device

        anno_down = F.interpolate(
            anno.float().unsqueeze(1), size=(h, w), mode='nearest'
        ).long().squeeze(1)

        # ── Step 1: match on final layer ──────────────────────────────────
        indices = []
        for b in range(B):
            segments = self._build_gt(anno_down[b])
            if not segments:
                indices.append(([], [], [], None))
                continue
            gt_classes = [s[0] for s in segments]
            gt_masks_b = torch.stack([s[1] for s in segments]).to(dev)
            cost       = self._cost_matrix(final_cls[b], final_masks[b],
                                           gt_classes, gt_masks_b)
            row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy())
            indices.append((row_ind, col_ind, gt_classes, gt_masks_b))

        # ── Step 2: final layer loss ──────────────────────────────────────
        final_loss = self._layer_loss_with_indices(
            final_cls, final_masks, anno, indices)

        if len(all_class_logits) <= 1:
            return final_loss

        # ── Step 3: aux losses with the same indices ──────────────────────
        aux_losses = [
            self._layer_loss_with_indices(cls_l, mask_l, anno, indices)
            for cls_l, mask_l in zip(all_class_logits[:-1], all_masks[:-1])
        ]
        return final_loss + self.aux_weight * torch.stack(aux_losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerFusion(nn.Module):
    """Mask2Former with camera-lidar fusion backbone.

    Returns (None, segmap, all_class_logits, all_masks) — same signature as
    MaskFormerFusion so the training engine requires no changes.

    Architectural alignment with the official Mask2Former:
      - Pixel decoder = MSDeformAttnPixelDecoder (deformable self-attention
        encoder, 6 layers by default) instead of a plain FPN.
      - Mask embedding MLP has 3 layers (paper §A.2).
      - No FCN shortcut head — matches true Mask2Former.
      - Aux losses reuse final-layer Hungarian matching indices.
    """

    def __init__(self,
                 backbone:               str  = 'swin_base_patch4_window7_224',
                 num_classes:            int  = 4,
                 pixel_decoder_channels: int  = 256,
                 transformer_d_model:    int  = 256,
                 num_queries:            int  = 100,
                 num_decoder_layers:     int  = 9,
                 n_encoder_layers:       int  = 6,
                 pretrained:             bool = True):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        bb_channels = self.backbone.feature_info.channels()

        # ── Residual fusion units (per scale, per modality) ───────────────────
        self.fusion_res_rgb   = nn.ModuleList([ResidualConvUnit(c) for c in bb_channels])
        self.fusion_res_lidar = nn.ModuleList([ResidualConvUnit(c) for c in bb_channels])

        # ── MSDeformAttn pixel decoder (official Mask2Former) ─────────────────
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_channels_list=bb_channels,
            out_channels=pixel_decoder_channels,
            n_encoder_layers=n_encoder_layers,
        )

        self.pixel_proj = (
            nn.Conv2d(pixel_decoder_channels, transformer_d_model, 1)
            if pixel_decoder_channels != transformer_d_model else nn.Identity()
        )

        num_scales = max(1, len(bb_channels) - 1)
        self.scale_proj = nn.ModuleList([
            nn.Conv2d(pixel_decoder_channels, transformer_d_model, 1)
            if pixel_decoder_channels != transformer_d_model else nn.Identity()
            for _ in range(num_scales)
        ])

        # ── Masked-attention decoder ──────────────────────────────────────────
        self.transformer_decoder = Mask2FormerDecoder(
            d_model=transformer_d_model,
            nhead=8,
            num_layers=num_decoder_layers,
            num_queries=num_queries,
            num_classes=num_classes,
            num_scales=num_scales,
        )

        # ── Direct pixel classification head (FCN shortcut) ───────────────────
        # Mirrors MaskFormerFusion: provides reliable gradient signal from
        # epoch 0 while the masked-attention queries bootstrap.  Without this,
        # the Q=100 softmax weighting is nearly uniform early in training,
        # collapsing segmap to a background-dominated prediction.
        self.direct_head = nn.Conv2d(transformer_d_model, num_classes, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_bchw(f: torch.Tensor) -> torch.Tensor:
        if f.ndim == 4 and f.shape[1] < f.shape[-1]:
            return f.permute(0, 3, 1, 2).contiguous()
        return f

    def _extract_features(self, rgb, lidar, modal):
        if modal == 'rgb':
            return [self.fusion_res_rgb[i](self._to_bchw(f))
                    for i, f in enumerate(self.backbone(rgb))]
        elif modal == 'lidar':
            return [self.fusion_res_lidar[i](self._to_bchw(f))
                    for i, f in enumerate(self.backbone(lidar))]
        elif modal in ('fusion', 'cross_fusion'):
            raw_rgb   = [self._to_bchw(f) for f in self.backbone(rgb)]
            raw_lidar = [self._to_bchw(f) for f in self.backbone(lidar)]
            return [
                self.fusion_res_rgb[i](fr) + self.fusion_res_lidar[i](fl)
                for i, (fr, fl) in enumerate(zip(raw_rgb, raw_lidar))
            ]
        else:
            raise ValueError(f"Unknown modal: {modal!r}")

    def forward(self,
                rgb:   torch.Tensor,
                lidar: torch.Tensor,
                modal: str = 'fusion'):
        """
        Returns:
            (None, segmap, all_class_logits, all_masks)
        """
        H, W = rgb.shape[-2], rgb.shape[-1]

        # 1. Backbone + fusion → list of multi-scale features
        features = self._extract_features(rgb, lidar, modal)

        # 2. Multi-scale pixel decoder
        multi_scale_raw, pixel_features_raw = self.pixel_decoder(features)

        # 3. Project to transformer d_model
        pixel_features = self.pixel_proj(pixel_features_raw)   # [B, d, h0, w0]

        # Project each multi-scale level, padding list to expected num_scales
        num_scales = len(self.scale_proj)
        if len(multi_scale_raw) >= num_scales:
            multi_scale = [self.scale_proj[i](s)
                           for i, s in enumerate(multi_scale_raw[:num_scales])]
        else:
            # Fewer FPN levels than expected: pad by repeating the last
            multi_scale = [self.scale_proj[i](multi_scale_raw[min(i, len(multi_scale_raw)-1)])
                           for i in range(num_scales)]

        # 4. Masked-attention transformer decoder (deep supervision)
        all_class_logits, all_masks = self.transformer_decoder(
            multi_scale_features=multi_scale,
            pixel_features=pixel_features,
        )

        # 5. Merge final layer predictions → dense segmentation (MaskFormer §3.3)
        cls_probs  = F.softmax(all_class_logits[-1], dim=-1)[..., :self.num_classes]
        mask_probs = torch.sigmoid(all_masks[-1])
        segmap     = torch.einsum('bqc,bqhw->bchw', cls_probs, mask_probs)

        # 6. Upsample to input resolution
        segmap = F.interpolate(segmap, size=(H, W), mode='bilinear', align_corners=False)

        return None, segmap, all_class_logits, all_masks
