#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Model evaluation metrics Python scripts

Created on June 18th, 2021
'''
import torch

def find_overlap_exclude_bg_ignore(n_classes, output, anno):
    """
    Correct IoU calculation that excludes background class (0) 
    WITHOUT modifying predictions based on ground truth.

    :param n_classes: Total number of classes (including background)
    :param output: Model output batch (B, C, H, W)
    :param anno: Ground truth batch (B, H, W)
    :return: area_overlap, area_pred, area_label, area_union
    """
    # Argmax over channels → predicted class per pixel
    _, pred_indices = torch.max(output, dim=1)

    # Compute overlap (TP): prediction matches annotation
    overlap = pred_indices * (pred_indices == anno).long()

    # Number of foreground classes (1..n_classes-1)
    num_eval_classes = n_classes - 1

    # Histogram for classes 1..n_classes-1
    area_overlap = torch.histc(overlap.float(),
                               bins=num_eval_classes,
                               min=0.5,
                               max=n_classes - 0.5)

    area_pred = torch.histc(pred_indices.float(),
                            bins=num_eval_classes,
                            min=0.5,
                            max=n_classes - 0.5)

    area_label = torch.histc(anno.float(),
                             bins=num_eval_classes,
                             min=0.5,
                             max=n_classes - 0.5)

    # Union = TP + FP + FN
    area_union = area_pred + area_label - area_overlap
    area_union = torch.clamp(area_union, min=1e-6)

    return area_overlap, area_pred, area_label, area_union
