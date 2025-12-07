"""
Comprehensive Evaluation Script for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script provides comprehensive evaluation of CycleGAN models.
"""

import os
import torch
from config import get_config
from data import UnalignedDataset
from models import CycleGANModel
from torch.utils.data import DataLoader
from metrics import MetricsCalculator
from utils.visualization import save_image, tensor2im
import json


def evaluate_model(opt, model_path=None):
    """
    Evaluate CycleGAN model comprehensively.
    
    Args:
        opt: Configuration object
        model_path: Path to model checkpoint (optional)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    opt.phase = 'test'
    dataset = UnalignedDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_threads)
    
    # Create model
    opt.isTrain = False
    model = CycleGANModel(opt)
    
    # Load model if path provided
    if model_path:
        model.load_networks(model_path)
    
    model.eval()
    model.to(device)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(device=device)
    
    # Collect images for evaluation
    real_A_images = []
    fake_B_images = []
    real_B_images = []
    fake_A_images = []
    
    print("Generating images for evaluation...")
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= opt.num_test:
                break
            
            model.set_input(data)
            model.test()
            
            visuals = model.get_current_visuals()
            
            real_A_images.append(visuals['real_A'].cpu())
            fake_B_images.append(visuals['fake_B'].cpu())
            real_B_images.append(visuals['real_B'].cpu())
            fake_A_images.append(visuals['fake_A'].cpu())
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = {}
    
    # A to B metrics
    print("Calculating A->B metrics...")
    real_A_tensor = torch.cat(real_A_images, dim=0)
    fake_B_tensor = torch.cat(fake_B_images, dim=0)
    real_B_tensor = torch.cat(real_B_images, dim=0)
    
    a2b_metrics = metrics_calc.calculate_all_metrics(
        real_images=real_B_tensor,
        fake_images=fake_B_tensor
    )
    metrics['A_to_B'] = a2b_metrics
    
    # B to A metrics
    print("Calculating B->A metrics...")
    b2a_metrics = metrics_calc.calculate_all_metrics(
        real_images=real_A_tensor,
        fake_images=fake_A_tensor
    )
    metrics['B_to_A'] = b2a_metrics
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for direction, metric_dict in metrics.items():
        print(f"\n{direction}:")
        for metric_name, value in metric_dict.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: N/A")
    
    # Save results
    results_dir = os.path.join(opt.results_dir, opt.name)
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_metrics = {}
        for direction, metric_dict in metrics.items():
            json_metrics[direction] = {
                k: float(v) if v is not None else None
                for k, v in metric_dict.items()
            }
        json.dump(json_metrics, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return metrics


if __name__ == '__main__':
    opt = get_config()
    opt.isTrain = False
    opt.phase = 'test'
    opt.results_dir = './results'
    
    # Evaluate
    metrics = evaluate_model(opt)
    
    print("\nEvaluation complete!")

