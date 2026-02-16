#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to merge 3 specialized CLFT checkpoints into a single checkpoint.
Creates a pseudo-merged ensemble containing all three specialized models in one file.
"""
import os
import torch
import argparse
from pathlib import Path


def load_checkpoint(checkpoint_path):
    """Load a PyTorch checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return checkpoint


def merge_checkpoints(vehicle_ckpt, sign_ckpt, human_ckpt, output_path):
    """Create a pseudo-merged checkpoint containing all 3 specialized models."""

    # Load all checkpoints
    vehicle_ckpt_data = load_checkpoint(vehicle_ckpt)
    sign_ckpt_data = load_checkpoint(sign_ckpt)
    human_ckpt_data = load_checkpoint(human_ckpt)

    # Create a combined checkpoint that stores all 3 models
    merged_checkpoint = {
        'ensemble_models': {
            'vehicle': vehicle_ckpt_data,
            'sign': sign_ckpt_data,
            'human': human_ckpt_data
        },
        'merged_from': ['vehicle_only', 'sign_only', 'human_only'],
        'merge_method': 'pseudo_merged_ensemble',
        'description': 'Single file containing all 3 specialized models for unified loading'
    }

    # Save merged checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged_checkpoint, output_path)
    print(f"Saved pseudo-merged checkpoint to: {output_path}")
    print("This file contains all 3 specialized models for unified ensemble prediction")

    return merged_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Merge 3 specialized CLFT checkpoints')
    parser.add_argument('--vehicle', type=str, required=True,
                       help='Path to vehicle-only checkpoint')
    parser.add_argument('--sign', type=str, required=True,
                       help='Path to sign-only checkpoint')
    parser.add_argument('--human', type=str, required=True,
                       help='Path to human-only checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for merged checkpoint')

    args = parser.parse_args()

    print("Merging CLFT checkpoints...")
    print(f"Vehicle: {args.vehicle}")
    print(f"Sign: {args.sign}")
    print(f"Human: {args.human}")
    print(f"Output: {args.output}")

    merged_ckpt = merge_checkpoints(args.vehicle, args.sign, args.human, args.output)

    print("✓ Checkpoint merging complete!")
    print(f"Merged from: {merged_ckpt['merged_from']}")
    print(f"Merge method: {merged_ckpt['merge_method']}")


if __name__ == '__main__':
    main()