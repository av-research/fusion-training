#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Model Benchmarking Script

This script benchmarks the pseudo-merged ensemble model across different modalities
and hardware to provide comprehensive performance metrics.
"""
import os
import json
import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import psutil
import GPUtil
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Try to import FLOPS calculation libraries
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with: pip install thop")

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not available. Install with: pip install fvcore")


class EnsembleBenchmarker:
    """Comprehensive benchmarking for ensemble models."""

    def __init__(self, device='auto'):
        """
        Initialize benchmarker.

        Args:
            device: Device to use ('auto', 'cpu', 'cuda', or specific GPU)
        """
        self.device = self._setup_device(device)
        self.results = []

    def _setup_device(self, device):
        """Setup the computation device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cpu':
            return torch.device('cpu')
        elif device.startswith('cuda'):
            if torch.cuda.is_available():
                return torch.device(device)
            else:
                print("CUDA not available, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _create_dummy_input(self):
        """Create dummy input tensors for ensemble benchmarking."""
        # ZOD dataset uses 384x384 images
        rgb = torch.randn(1, 3, 384, 384)
        lidar = torch.randn(1, 3, 384, 384)
        return rgb, lidar

    def _count_ensemble_parameters(self, predictor):
        """Count total parameters in ensemble models."""
        if hasattr(predictor, 'models') and predictor.models:
            # Pseudo-merged ensemble
            total_params = 0
            trainable_params = 0
            for model in predictor.models:
                total_params += sum(p.numel() for p in model.parameters())
                trainable_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
        elif hasattr(predictor, 'model'):
            # Single model fallback
            total_params = sum(p.numel() for p in predictor.model.parameters())
            trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
        else:
            return {
                'total_parameters': 0,
                'trainable_parameters': 0,
                'total_parameters_m': 0,
                'trainable_parameters_m': 0
            }

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_parameters_m': total_params / 1e6,
            'trainable_parameters_m': trainable_params / 1e6
        }

    def _calculate_ensemble_flops(self, predictor, modality='cross_fusion'):
        """Calculate FLOPS for ensemble predictor."""
        rgb, lidar = self._create_dummy_input()

        # Move inputs to same device as model
        rgb = rgb.to(self.device)
        lidar = lidar.to(self.device)

        flops_info = {
            'flops_available': False,
            'total_flops': None,
            'flops_giga': None,
            'flops_method': None
        }

        # For ensemble, we need to calculate FLOPS for each model
        if hasattr(predictor, 'models') and predictor.models:
            total_flops = 0
            method_used = None

            for i, model in enumerate(predictor.models):
                model_flops = self._calculate_single_model_flops(model, rgb, lidar, modality)
                if model_flops['flops_available']:
                    total_flops += model_flops['total_flops']
                    if method_used is None:
                        method_used = model_flops['flops_method']

            if total_flops > 0:
                flops_info.update({
                    'flops_available': True,
                    'total_flops': total_flops,
                    'flops_giga': total_flops / 1e9,
                    'flops_method': method_used or 'ensemble'
                })
        elif hasattr(predictor, 'model'):
            # Single model fallback
            flops_info = self._calculate_single_model_flops(predictor.model, rgb, lidar, modality)

        return flops_info

    def _calculate_single_model_flops(self, model, rgb, lidar, modality):
        """Calculate FLOPS for a single CLFT model."""
        flops_info = {
            'flops_available': False,
            'total_flops': None,
            'flops_method': None
        }

        # Try thop first
        if THOP_AVAILABLE:
            try:
                flops, params = profile(model, inputs=(rgb, lidar, modality), verbose=False)
                flops_info.update({
                    'flops_available': True,
                    'total_flops': flops,
                    'flops_method': 'thop'
                })
            except Exception as e:
                print(f"thop profiling failed: {e}")

        # Try fvcore as fallback
        if not flops_info['flops_available'] and FVCORE_AVAILABLE:
            try:
                flop_analyzer = FlopCountAnalysis(model, (rgb, lidar, modality))
                total_flops = flop_analyzer.total()
                flops_info.update({
                    'flops_available': True,
                    'total_flops': total_flops,
                    'flops_method': 'fvcore'
                })
            except Exception as e:
                print(f"fvcore profiling failed: {e}")

        return flops_info

    def _measure_ensemble_inference_time(self, predictor, modality='cross_fusion', num_runs=10, warmup_runs=2):
        """Measure inference time for ensemble on current device."""
        rgb, lidar = self._create_dummy_input()

        # Move inputs to device
        rgb = rgb.to(self.device)
        lidar = lidar.to(self.device)

        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = predictor.predict(rgb, lidar, modal=modality)

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = predictor.predict(rgb, lidar, modal=modality)
                end_time = time.time()
                times.append(end_time - start_time)

        times = np.array(times)

        result = {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'num_runs': num_runs
        }

        return result

    def _get_system_info(self):
        """Get system and hardware information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }

        if torch.cuda.is_available():
            gpu_info = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu_info:
                info.update({
                    'gpu_name': gpu_info.name,
                    'gpu_memory_total_gb': gpu_info.memoryTotal / 1024,
                    'gpu_driver': torch.version.cuda
                })

        return info

    def benchmark_ensemble(self, modality='cross_fusion'):
        """Benchmark the pseudo-merged ensemble."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: ZOD Pseudo-Merged Ensemble")
        print(f"{'='*60}")

        # Load ensemble predictor
        from zod_ensemble_predictor import ZODMergedEnsemblePredictor
        predictor = ZODMergedEnsemblePredictor(device=self.device)

        # Get model info
        model_info = self._count_ensemble_parameters(predictor)
        model_info.update({
            'model_name': 'ZOD Pseudo-Merged Ensemble',
            'backbone': 'clft_ensemble',
            'dataset': 'zod',
            'image_size': 384,
            'pretrained': True,
            'num_models': len(predictor.models) if hasattr(predictor, 'models') else 1
        })

        print(f"\nTesting modality: {modality.upper()}")

        # Calculate FLOPS
        flops_info = self._calculate_ensemble_flops(predictor, modality)

        # Measure inference time
        timing_info = self._measure_ensemble_inference_time(predictor, modality, num_runs=10, warmup_runs=2)

        # Combine results
        result = {
            'model_name': 'ZOD Pseudo-Merged Ensemble',
            'modality': modality,
            **model_info,
            **flops_info,
            **timing_info,
            'device': str(self.device),
            'device_type': self.device.type
        }

        self.results.append(result)

        # Print summary
        print(f"  Models: {model_info['num_models']}")
        print(f"  Parameters: {model_info['total_parameters_m']:.1f}M")
        if flops_info['flops_available']:
            print(f"  FLOPS: {flops_info['flops_giga']:.2f}G ({flops_info['flops_method']})")
        print(f"  Inference: {timing_info['mean_time_ms']:.2f}±{timing_info['std_time_ms']:.2f}ms")
        print(f"  FPS: {timing_info['fps']:.1f}")
        if 'gpu_memory_mean_mb' in timing_info:
            print(f"  GPU Memory: {timing_info['gpu_memory_mean_mb']:.1f}±{timing_info['gpu_memory_std_mb']:.1f}MB")
        if 'ram_memory_mean_mb' in timing_info:
            print(f"  RAM Memory: {timing_info['ram_memory_mean_mb']:.1f}±{timing_info['ram_memory_std_mb']:.1f}MB")

    def save_results(self, output_path=None):
        """Save results to JSON file."""
        if output_path is None:
            # Create timestamped filename in ensemble benchmark directory
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            benchmark_dir = '/media/tom/ml/projects/fusion-training/logs/zod/clft/specialization/ensemble/benchmark'
            os.makedirs(benchmark_dir, exist_ok=True)
            output_path = os.path.join(benchmark_dir, f'benchmark_results_{timestamp}.json')

        # Add system info to results
        system_info = self._get_system_info()

        # For ensemble, we don't have epoch info, so set to None
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': None,
            'epoch_uuid': None,
            'training_uuid': None,
            'system_info': system_info,
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        
        # Send results to vision service
        try:
            from integrations.vision_service import send_benchmark_results_from_file
            
            success = send_benchmark_results_from_file(output_path)
            if success:
                print("Benchmark results successfully sent to vision service")
            else:
                print("Failed to send benchmark results to vision service")
        except ImportError:
            print("Warning: vision_service module not found, skipping upload to vision service")
        except Exception as e:
            print(f"Error sending benchmark results to vision service: {e}")

    def create_summary_table(self, output_path=None):
        """Create a summary table of results."""
        if not self.results:
            print("No results to summarize")
            return

        if output_path is None:
            benchmark_dir = '/media/tom/ml/projects/fusion-training/logs/zod/clft/specialization/ensemble/benchmark'
            os.makedirs(benchmark_dir, exist_ok=True)
            output_path = os.path.join(benchmark_dir, 'benchmark_summary.csv')

        df = pd.DataFrame(self.results)

        # Create summary with key metrics
        summary_cols = [
            'model_name', 'modality', 'device_type',
            'num_models', 'total_parameters_m', 'mean_time_ms', 'fps'
        ]

        # Add FLOPS column if available
        if 'flops_giga' in df.columns:
            summary_cols.insert(5, 'flops_giga')

        summary_df = df[summary_cols].copy()
        summary_df = summary_df.round(3)
        summary_df.to_csv(output_path, index=False)

        print(f"\nSummary table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Ensemble Model Benchmarking')
    parser.add_argument('-d', '--device', default='cuda',
                       help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    parser.add_argument('-m', '--modality', default='cross_fusion',
                       help='Fusion modality to test (rgb, lidar, cross_fusion)')
    parser.add_argument('--single', action='store_true',
                       help='Benchmark on single device only (default: benchmark on both CPU and GPU if available)')
    parser.add_argument('--include-cpu', action='store_true',
                       help='Include CPU benchmarking for ensemble (may have loading issues)')

    args = parser.parse_args()

    devices_to_test = []

    # For ensemble, default to CUDA only since CPU loading may have issues
    # But allow CPU if explicitly requested
    if not args.single and torch.cuda.is_available() and args.device != 'cpu':
        devices_to_test = ['cuda']
        if args.include_cpu:
            devices_to_test.append('cpu')
        print(f"Benchmarking ensemble on {', '.join(d.upper() for d in devices_to_test)} (ensemble optimized for GPU)")
    else:
        devices_to_test = [args.device]
        device_name = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Benchmarking ensemble on single device: {device_name}")

    all_results = []

    for device in devices_to_test:
        print(f"\n{'='*80}")
        print(f"Running ensemble benchmarks on {device.upper()}")
        print(f"{'='*80}")

        # Create benchmarker for this device
        benchmarker = EnsembleBenchmarker(device)

        # Benchmark ensemble
        benchmarker.benchmark_ensemble(args.modality)

        # Collect results
        all_results.extend(benchmarker.results)

    # Create combined benchmarker for saving
    combined_benchmarker = EnsembleBenchmarker(devices_to_test[0])
    combined_benchmarker.results = all_results

    # Save combined results
    combined_benchmarker.save_results()


if __name__ == '__main__':
    main()