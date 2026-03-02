#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskFormer-based fusion model for camera-lidar segmentation.

Pure-PyTorch implementation faithful to Cheng et al.,
"Per-Pixel Classification is Not All You Need for Semantic Segmentation",
NeurIPS 2021.  Reference: facebookresearch/MaskFormer.

Architecture:
  1. Shared backbone (timm) extracts multi-scale features.
  2. PixelDecoder (FPN-style) produces rich pixel-level feature maps.
  3. TransformerDecoder cross-attends N learnable object queries against the
     pixel features.  ALL 6 intermediate layer outputs are returned for
     auxiliary (deep) supervision, matching deep_supervision=True in the
     official TransformerPredictor.
  4. At inference the final-layer query outputs are merged into a dense
     segmentation map (official semantic_inference formula):
       segmap[c] = Σ_q  softmax(class_logits_q)[c]  ×  sigmoid(mask_q)
     where the no-object column is simply dropped, not used to gate.

Loss (MaskFormerCriterion / SetCriterion):
  Matching cost : class CE + focal-loss (mask) + dice (mask)
  Training loss : weight_class × CE  +  weight_mask × focal  +  weight_dice × dice
  Applied to final AND all auxiliary decoder layer outputs.
  Default weights match the paper (class=2, mask=5, dice=5, eos=0.1).

Known deviation from original:
  The pixel decoder is a plain FPN without the transformer encoder layer
  (TransformerEncoderPixelDecoder) used in the official Swin configs.
  This is a deliberate simplification for the small-dataset fusion setting.

Camera-lidar fusion is handled by element-wise addition of rgb and lidar
backbone features (early-fusion variant), identical to other models in this
codebase.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from scipy.optimize import linear_sum_assignment

# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers — faithful to facebookresearch/MaskFormer criterion.py
# (Cheng et al., "Per-Pixel Classification is Not All You Need", NeurIPS 2021)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid_focal_loss(inputs: torch.Tensor,
                        targets: torch.Tensor,
                        num_masks: float,
                        alpha: float = 0.25,
                        gamma: float = 2.0) -> torch.Tensor:
    """Sigmoid focal loss normalised by num_masks."""
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t     = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss    = ce_loss * ((1.0 - p_t) ** gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * loss).mean(1).sum() / num_masks


def _dice_loss(inputs: torch.Tensor,
               targets: torch.Tensor,
               num_masks: float) -> torch.Tensor:
    """Dice loss normalised by num_masks."""
    p   = inputs.sigmoid().flatten(1)
    t   = targets.flatten(1)
    num = 2.0 * (p * t).sum(-1)
    den = p.sum(-1) + t.sum(-1)
    return (1.0 - (num + 1.0) / (den + 1.0)).sum() / num_masks


@torch.no_grad()
def _batch_focal_cost(pred_flat: torch.Tensor,
                      gt_flat:   torch.Tensor,
                      alpha: float = 0.25,
                      gamma: float = 2.0) -> torch.Tensor:
    """Vectorised focal-loss cost matrix [Q, M] for Hungarian matching.

    pred_flat : [Q, HW]  raw logits
    gt_flat   : [M, HW]  binary float masks
    """
    prob = pred_flat.sigmoid()
    fp   = alpha * ((1.0 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
               pred_flat, torch.ones_like(pred_flat), reduction='none')   # [Q, HW]
    fn   = (1.0 - alpha) * (prob ** gamma) * F.binary_cross_entropy_with_logits(
               pred_flat, torch.zeros_like(pred_flat), reduction='none')  # [Q, HW]
    hw   = pred_flat.shape[1]
    return (torch.einsum('qn,mn->qm', fp, gt_flat) +
            torch.einsum('qn,mn->qm', fn, 1.0 - gt_flat)) / hw


class ResidualConvUnit(nn.Module):
    """3×3 residual conv block — matches swin_transformer_fusion.py."""

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


class PixelDecoder(nn.Module):
    """FPN-style pixel decoder (multi-scale → single high-res feature map).

    Takes the list of backbone feature maps (from finest to coarsest resolution)
    and produces a single feature map at the finest resolution via lateral
    connections and top-down upsampling.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int = 256):
        super().__init__()

        # 1×1 lateral projections for each scale
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )
        # Output refinement
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of [B, C_i, H_i, W_i], ordered fine→coarse.
        Returns:
            [B, out_channels, H_0, W_0] at the finest resolution.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down path: start from coarsest, upsample and add
        out = laterals[-1]
        for lateral in reversed(laterals[:-1]):
            out = F.interpolate(out, size=lateral.shape[2:],
                                mode='bilinear', align_corners=False)
            out = out + lateral

        return self.output_conv(out)


class TransformerDecoder(nn.Module):
    """Per-object-query transformer decoder (MaskFormer §3.2).

    N learnable query embeddings cross-attend to the pixel-level memory
    produced by the PixelDecoder.  Each query predicts:
      - a class distribution over (num_classes + 1) labels (the extra label is
        the "∅ / no-object" token)
      - a binary foreground mask over the spatial grid

    The masks are computed as the dot product between the query's mask
    embedding and every pixel's feature vector, following the original paper.
    """

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_queries: int = 100,
                 num_classes: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Learned 2-D positional encoding for the pixel-feature memory.
        # Without this the cross-attention has no spatial signal and queries
        # cannot learn WHERE to attend, so masks stay unlocalized.
        # Sized for the coarsest expected feature map (backbone/4 = 64 for
        # a 256-input swin), but dynamically interpolated at forward time.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, d_model, 64, 64))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Standard PyTorch transformer decoder.
        # dropout=0.0 matches Mask2Former practice and eliminates the train/eval
        # behavioural gap: with dropout>0 the model trains under stochastic mask
        # suppression but is evaluated deterministically, which consistently
        # inflates background query contributions at inference and collapses mIoU.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Classification head: num_classes + 1 (no-object)
        self.class_embed = nn.Linear(d_model, num_classes + 1)

        # Mask embedding: projects query features → spatial mask logits
        self.mask_embed = nn.Linear(d_model, d_model)

    def forward(self, pixel_features: torch.Tensor):
        """
        Runs all decoder layers and returns outputs from every layer for
        auxiliary (deep) supervision, matching the official MaskFormer
        TransformerPredictor (deep_supervision=True).

        Args:
            pixel_features: [B, C, H, W]  (output of PixelDecoder)
        Returns:
            all_class_logits : list[Tensor[B, Q, C+1]], length = num_layers
            all_masks        : list[Tensor[B, Q, H, W]], length = num_layers
            Index -1 is the final (main) output; 0..−2 are auxiliary.
        """
        b, c, h, w = pixel_features.shape

        # Add 2-D positional encoding (interpolated to actual feature map size)
        pos = F.interpolate(self.pos_embed, size=(h, w),
                            mode='bilinear', align_corners=False)  # [1, C, h, w]
        memory_with_pos = (pixel_features + pos).flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Broadcast queries over the batch
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, Q, C]

        # Iterate through decoder layers, collecting per-layer predictions for
        # auxiliary losses (deep supervision) — identical to the official
        # MaskFormer TransformerPredictor with deep_supervision=True.
        all_class_logits: list[torch.Tensor] = []
        all_masks:        list[torch.Tensor] = []

        for layer in self.decoder.layers:
            q = layer(q, memory_with_pos)

            # Per-query class logits and mask dot-product for this layer
            cl = self.class_embed(q)                                         # [B, Q, C+1]
            mf = self.mask_embed(q)                                          # [B, Q, C]
            m  = torch.einsum('bqc,bchw->bqhw', mf,
                              pixel_features) / math.sqrt(self.d_model)      # [B, Q, h, w]
            all_class_logits.append(cl)
            all_masks.append(m)

        return all_class_logits, all_masks


class MaskFormerCriterion(nn.Module):
    """Bipartite matching loss for MaskFormer.

    Faithful to facebookresearch/MaskFormer SetCriterion.  For each image:
      1. Extract GT segments from the semantic annotation map.
      2. Build a [Q × M] cost matrix (class CE + focal mask + dice mask).
      3. Run Hungarian algorithm to find optimal query → GT assignment.
      4. Classification CE over ALL queries (matched → GT class,
         unmatched → no-object), normalised by Q via F.cross_entropy.
      5. Sigmoid focal loss + Dice loss over matched pairs,
         normalised by the total number of matched GT segments.

    Deep-supervision: when aux_outputs are provided (list of
    (class_logits, masks) from intermediate decoder layers), the same loss
    formula is applied to each auxiliary output and the results are summed.

    Loss weights (paper §A.2 Table 8):
      cost  : class=1, mask=20, dice=1
      loss  : class=2, mask=5,  dice=5
      eos_coef (no-object weight in class CE) = 0.1
    """

    def __init__(self,
                 num_classes: int,
                 cost_class: float = 1.0,
                 cost_mask: float = 20.0,   # paper §A.2: λ_bce=20 in cost matrix
                 cost_dice: float = 1.0,    # paper §A.2: λ_dice=1 in cost matrix
                 weight_class: float = 2.0,
                 weight_mask: float = 5.0,
                 weight_dice: float = 5.0,
                 no_object_coef: float = 0.1,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.num_classes    = num_classes
        self.cost_class     = cost_class
        self.cost_mask      = cost_mask
        self.cost_dice      = cost_dice
        self.weight_class   = weight_class
        self.weight_mask    = weight_mask
        self.weight_dice    = weight_dice
        self.no_object_coef = no_object_coef

        # CE weight tensor: [class_0_w, …, class_{C-1}_w, no_object_w]
        # Original MaskFormer (Cheng et al., NeurIPS 2021) applies eos_coef as
        # the weight of the no-object class in a single weight vector used for
        # ALL queries — this is different from multiplying the entire loss by the
        # scalar.  Always build this vector so the paper's semantics are matched
        # even when no per-class weights are supplied.
        if class_weights is not None:
            eos = torch.cat([class_weights,
                             class_weights.new_tensor([no_object_coef])])
        else:
            # Foreground classes all weight 1; no-object class weighted by eos_coef
            eos = torch.ones(num_classes + 1)
            eos[-1] = no_object_coef
        # register_buffer so it moves with .to(device) but is not a parameter
        self.register_buffer('ce_weight', eos)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _build_gt(self, anno_b: torch.Tensor):
        """Extract per-class binary masks from a [H, W] annotation map.

        Returns list of (class_idx: int, binary_mask: [H, W] float32).
        ALL classes including background (class 0) are included so that one
        query is explicitly trained to predict background.  Without this,
        the fg_prob gating in the segmap merging suppresses unmatched
        (no-object) query contributions to zero for every channel — including
        channel 0 — leaving background permanently unpredicted.  The result is
        near-zero precision despite high recall: argmax never picks class 0.
        Including background trains a dedicated query whose sigmoid mask covers
        the background region, giving channel 0 a proper positive signal.
        """
        segments = []
        for c in range(self.num_classes):   # include class 0 (background)
            mask = (anno_b == c).float()
            if mask.sum() > 0:
                segments.append((c, mask))
        return segments

    @torch.no_grad()
    def _cost_matrix(self,
                     cls_logits_q: torch.Tensor,   # [Q, C+1]
                     pred_masks_q: torch.Tensor,   # [Q, h, w]
                     gt_classes:   list[int],
                     gt_masks:     torch.Tensor,   # [M, h, w]
                     ) -> torch.Tensor:             # [Q, M]
        Q = cls_logits_q.shape[0]
        M = len(gt_classes)
        dev = cls_logits_q.device

        # ── Class cost: −p(gt_class) ──────────────────────────────────────────
        cls_probs   = F.softmax(cls_logits_q, dim=-1)          # [Q, C+1]
        gt_idx      = torch.tensor(gt_classes, device=dev)     # [M]
        class_cost  = -cls_probs[:, gt_idx]                    # [Q, M]

        # ── Mask costs (vectorised Q×M) ───────────────────────────────────────
        pred_flat = pred_masks_q.flatten(1)       # [Q, HW]
        pred_sig  = pred_flat.sigmoid()           # [Q, HW]
        gt_flat   = gt_masks.flatten(1)           # [M, HW]
        HW        = pred_flat.shape[1]

        # Mask cost: sigmoid focal loss — matches official HungarianMatcher
        # (facebookresearch/MaskFormer matcher.py: batch_sigmoid_focal_loss)
        focal_cost = _batch_focal_cost(pred_flat, gt_flat)             # [Q, M]

        # Dice cost
        num       = 2 * torch.einsum('qn,mn->qm', pred_sig, gt_flat)  # [Q, M]
        den       = pred_sig.sum(-1, keepdim=True) + gt_flat.sum(-1).unsqueeze(0) + 1e-5
        dice_cost = 1.0 - num / den                                    # [Q, M]

        return (self.cost_class * class_cost
                + self.cost_mask  * focal_cost
                + self.cost_dice  * dice_cost)

    # ── Forward ───────────────────────────────────────────────────────────────

    def _loss_single_output(
        self,
        class_logits: torch.Tensor,   # [B, Q, C+1]
        pred_masks:   torch.Tensor,   # [B, Q, h, w]
        anno_down:    torch.Tensor,   # [B, h, w]  long  (already downsampled)
    ) -> torch.Tensor:
        """Compute Hungarian-matching loss for one set of predictions.

        Called for both the final decoder output and each auxiliary output.
        Faithful to facebookresearch/MaskFormer SetCriterion.
        """
        B, Q, _ = class_logits.shape
        dev = class_logits.device
        no_obj_idx = torch.tensor([self.num_classes], dtype=torch.long, device=dev)

        total_loss  = class_logits.new_tensor(0.0)
        num_matched = 0  # total matched GT segments across batch (for mask normalisation)

        # Collect matched-pair mask tensors for batch-normalised focal+dice
        all_pm: list[torch.Tensor] = []
        all_gm: list[torch.Tensor] = []

        for b in range(B):
            segments = self._build_gt(anno_down[b])

            if not segments:
                # Degenerate: push all queries to no-object
                total_loss = total_loss + F.cross_entropy(
                    class_logits[b], no_obj_idx.expand(Q),
                    weight=self.ce_weight,
                )
                continue

            gt_classes = [s[0] for s in segments]
            gt_masks   = torch.stack([s[1] for s in segments]).to(dev)  # [M, h, w]

            # ── Hungarian matching ────────────────────────────────────────────
            cost = self._cost_matrix(
                class_logits[b], pred_masks[b], gt_classes, gt_masks)
            row_ind, col_ind = linear_sum_assignment(
                cost.cpu().detach().numpy())
            matched = set(row_ind.tolist())
            num_matched += len(row_ind)

            # ── Classification loss (official: single CE over all queries) ────
            # Build a target-class vector: matched → GT class, rest → no-object.
            tgt_classes = torch.full((Q,), self.num_classes,
                                     dtype=torch.long, device=dev)
            for r, c in zip(row_ind, col_ind):
                tgt_classes[r] = gt_classes[c]
            total_loss = total_loss + self.weight_class * F.cross_entropy(
                class_logits[b], tgt_classes, weight=self.ce_weight,
            )

            # ── Collect matched mask pairs (focal + dice computed below) ──────
            for r, c in zip(row_ind, col_ind):
                all_pm.append(pred_masks[b, r].flatten().unsqueeze(0))  # [1, HW]
                all_gm.append(gt_masks[c].flatten().unsqueeze(0))       # [1, HW]

        # ── Mask losses: sigmoid focal + dice, normalised by num_matched ──────
        # Matches facebookresearch/MaskFormer criterion.py loss_masks():
        #   loss_mask = sigmoid_focal_loss(src, tgt, num_masks)
        #   loss_dice = dice_loss(src, tgt, num_masks)
        if all_pm:
            n = float(max(1, num_matched))
            pm_cat = torch.cat(all_pm, dim=0)  # [num_matched, HW]
            gm_cat = torch.cat(all_gm, dim=0)  # [num_matched, HW]
            total_loss = total_loss + self.weight_mask * _sigmoid_focal_loss(
                pm_cat, gm_cat, n)
            total_loss = total_loss + self.weight_dice * _dice_loss(
                pm_cat, gm_cat, n)

        return total_loss / B

    def forward(self,
                class_logits: torch.Tensor,            # [B, Q, C+1]
                pred_masks:   torch.Tensor,            # [B, Q, h, w]
                anno:         torch.Tensor,            # [B, H, W]  long
                aux_outputs:  list | None = None,      # [(cls, mask), ...] auxiliary layers
                ) -> torch.Tensor:
        """Compute total loss: final-layer loss + auxiliary-layer losses.

        aux_outputs is a list of (class_logits, masks) from intermediate
        transformer decoder layers (deep supervision / auxiliary losses),
        as produced by MaskFormerFusion.forward().
        """
        _, _, h, w = pred_masks.shape

        # Downsample annotation once to pixel-decoder resolution
        anno_down = F.interpolate(
            anno.float().unsqueeze(1), size=(h, w), mode='nearest'
        ).long().squeeze(1)  # [B, h, w]

        # Final-layer loss
        total_loss = self._loss_single_output(class_logits, pred_masks, anno_down)

        # Auxiliary losses (deep supervision) — same formula applied to each
        # intermediate decoder layer output, following the official implementation.
        if aux_outputs:
            for aux_cls, aux_mask in aux_outputs:
                # Aux masks may be at a different spatial resolution; re-downsample
                _, _, ah, aw = aux_mask.shape
                if (ah, aw) != (h, w):
                    ad = F.interpolate(
                        anno.float().unsqueeze(1), size=(ah, aw), mode='nearest'
                    ).long().squeeze(1)
                else:
                    ad = anno_down
                total_loss = total_loss + self._loss_single_output(
                    aux_cls, aux_mask, ad)

        return total_loss


class MaskFormerFusion(nn.Module):
    """MaskFormer with camera-lidar fusion backbone.

    Differences from the original MaskFormer:
    - Backbone is any timm model (not necessarily a Swin Transformer).
    - Camera and LiDAR streams share a single backbone; their features are
      fused by element-wise addition before the pixel decoder.
    - Returns (None, segmap) where segmap is a dense [B, C, H, W] logit
      tensor, so the model is a drop-in replacement for all other models in
      this codebase and works with the existing CrossEntropyLoss training
      engine.

    The dense segmap is produced by the standard MaskFormer inference step:
        segmap[:, c, :, :] = Σ_q  softmax(class_logits_q)[c] · sigmoid(mask_q)
    """

    def __init__(self,
                 backbone: str = 'swin_base_patch4_window7_224',
                 num_classes: int = 4,
                 pixel_decoder_channels: int = 256,
                 transformer_d_model: int = 256,
                 num_queries: int = 100,
                 pretrained: bool = True):
        super().__init__()

        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        bb_channels = self.backbone.feature_info.channels()

        # ── Residual fusion units (residual_average strategy) ─────────────────
        # One ResidualConvUnit per backbone scale, per modality stream.
        # Mirrors the design in swin_transformer_fusion.py:
        #   fused_i = res_rgb_i(feat_rgb_i) + res_lidar_i(feat_lidar_i) + prev
        self.fusion_res_rgb   = nn.ModuleList(
            [ResidualConvUnit(c) for c in bb_channels])
        self.fusion_res_lidar = nn.ModuleList(
            [ResidualConvUnit(c) for c in bb_channels])

        # ── Pixel Decoder (FPN) ───────────────────────────────────────────────
        self.pixel_decoder = PixelDecoder(bb_channels, pixel_decoder_channels)

        # Project pixel decoder output to transformer d_model if sizes differ
        self.pixel_proj = (
            nn.Conv2d(pixel_decoder_channels, transformer_d_model, 1)
            if pixel_decoder_channels != transformer_d_model else nn.Identity()
        )

        # ── Transformer Decoder ───────────────────────────────────────────────
        self.transformer_decoder = TransformerDecoder(
            d_model=transformer_d_model,
            nhead=8,   # matches MaskFormer paper (Cheng et al., NeurIPS 2021)
            num_layers=6,
            num_queries=num_queries,
            num_classes=num_classes,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _to_bchw(f: torch.Tensor) -> torch.Tensor:
        """Normalise a backbone feature map to [B, C, H, W]."""
        if f.ndim == 4 and f.shape[1] < f.shape[-1]:   # BHWC → BCHW
            return f.permute(0, 3, 1, 2).contiguous()
        return f

    def _extract_features(self, rgb: torch.Tensor,
                          lidar: torch.Tensor,
                          modal: str) -> list[torch.Tensor]:
        """Run backbone and fuse with residual_average strategy.

        Applies a ResidualConvUnit to each stream at each backbone scale,
        then sums the two streams.  Mirrors swin_transformer_fusion.py's
        residual_average, adapted for multi-scale features with varying
        channel widths (no cross-scale previous-stage threading: the FPN
        PixelDecoder already provides that cross-scale integration).

            fused_i = res_rgb_i(feat_rgb_i) + res_lidar_i(feat_lidar_i)
        """
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

    def forward(self, rgb: torch.Tensor,
                lidar: torch.Tensor,
                modal: str = 'fusion'):
        """
        Args:
            rgb   : [B, 3, H, W]
            lidar : [B, C_l, H, W]
            modal : 'rgb' | 'lidar' | 'fusion' | 'cross_fusion'
        Returns:
            (None, segmap)  where segmap is [B, num_classes, H, W]
        """
        H, W = rgb.shape[-2], rgb.shape[-1]

        # 1. Backbone + fusion
        features = self._extract_features(rgb, lidar, modal)

        # 2. FPN pixel decoder
        pixel_features = self.pixel_decoder(features)          # [B, pd_ch, h, w]
        pixel_features = self.pixel_proj(pixel_features)       # [B, d_model, h, w]

        # 3. Transformer decoder → per-layer class logits + masks (all layers)
        all_class_logits, all_masks = self.transformer_decoder(pixel_features)
        # each element: class_logits [B, Q, C+1], masks [B, Q, h, w]
        class_logits = all_class_logits[-1]   # final layer
        masks        = all_masks[-1]
        # Auxiliary outputs from intermediate layers (for deep supervision)
        aux_outputs = list(zip(all_class_logits[:-1], all_masks[:-1]))

        # 4. Merge — official MaskFormer semantic_inference formula
        #   (Cheng et al., NeurIPS 2021, mask_former_model.py):
        #
        #     semseg[c] = Σ_q  softmax(cls_logits)[q, c]  ×  sigmoid(mask)[q]
        #
        #   The no-object column is dropped (sliced off) but NOT used to gate the
        #   remaining class probabilities.  After training, matched queries have
        #   P(class_c) ≈ 1 and unmatched queries have P(no-obj) ≈ 1 so their
        #   P(class_c) ≈ 0, giving the same limiting behaviour without the extra
        #   multiplicative factor that deviates from the paper formula.
        cls_probs  = F.softmax(class_logits, dim=-1)[..., :self.num_classes]  # [B, Q, C]
        mask_probs = masks.sigmoid()                                           # [B, Q, h, w]
        segmap     = torch.einsum('bqc,bqhw->bchw', cls_probs, mask_probs)    # [B, C, h, w]

        # 5. Upsample to original input resolution
        segmap = F.interpolate(segmap, size=(H, W),
                               mode='bilinear', align_corners=False)

        return None, segmap, class_logits, masks, aux_outputs