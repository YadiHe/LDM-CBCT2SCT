#!/usr/bin/env python
"""
Training Loss Curves Visualization Tool

Usage:
  # View VAE training curves
  python view_training_loss.py --vae
  
  # View UNet training curves
  python view_training_loss.py --unet
  
  # View both
  python view_training_loss.py --both
  
  # Specify custom log file
  python view_training_loss.py --log-file path/to/your/training.log --title "My Training"
"""

import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import numpy as np
import os
import argparse
from datetime import datetime

def parse_log_file(log_file):
    """
    Parse training log file
    Returns: epochs, train_losses, val_losses, learning_rates, best_epochs
    """
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []
    best_epochs = []  # Record epochs where best model was saved
    
    if not os.path.exists(log_file):
        print(f"Error: Log file does not exist: {log_file}")
        return None, None, None, None, None
    
    print(f"Reading log file: {log_file}")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Match training loss line - 支持两种格式
            # 格式1 (UNet): Epoch X | Train Loss: Y | Val Loss: Z
            match1 = re.search(r'Epoch (\d+) \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
            # 格式2 (VAE): Epoch X, Train Loss: Y, Val Loss: Z, LR: W
            match2 = re.search(r'Epoch (\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+), LR: ([\d.e-]+)', line)
            
            if match1:
                epoch = int(match1.group(1))
                train_loss = float(match1.group(2))
                val_loss = float(match1.group(3))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                # UNet日志没有LR信息，使用默认值或前一个值
                if learning_rates:
                    learning_rates.append(learning_rates[-1])
                else:
                    learning_rates.append(1e-5)  # 默认学习率
                    
            elif match2:
                epoch = int(match2.group(1))
                train_loss = float(match2.group(2))
                val_loss = float(match2.group(3))
                lr = float(match2.group(4))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                learning_rates.append(lr)
            
            # Match best model save line
            best_match = re.search(r'Saved new best.*at epoch (\d+)', line)
            if best_match:
                best_epoch = int(best_match.group(1))
                best_epochs.append(best_epoch)
    
    if not epochs:
        print("Warning: No training data found")
        return None, None, None, None, None
    
    print(f"Successfully parsed {len(epochs)} epochs of data")
    return epochs, train_losses, val_losses, learning_rates, best_epochs


def plot_training_curves(log_file, save_path=None, title="Training Loss Curves"):
    """
    Plot training loss curves
    
    Args:
        log_file: Path to log file
        save_path: Path to save the plot (optional)
        title: Chart title
    """
    
    # Parse log
    epochs, train_losses, val_losses, learning_rates, best_epochs = parse_log_file(log_file)
    
    if epochs is None:
        return None
    
    # Statistics
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    best_epoch = epochs[np.argmin(val_losses)]
    current_epoch = epochs[-1]
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    
    print("\n" + "="*60)
    print("Training Statistics")
    print("="*60)
    print(f"Current Epoch: {current_epoch}")
    print(f"Best Val Loss: {min_val_loss:.6f} (Epoch {best_epoch})")
    print(f"Min Train Loss: {min_train_loss:.6f}")
    print(f"Current Train Loss: {final_train_loss:.6f}")
    print(f"Current Val Loss: {final_val_loss:.6f}")
    print(f"Best Model Saves: {len(best_epochs)}")
    print("="*60 + "\n")
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 12))
    
    # ============ Subplot 1: Training and Validation Loss ============
    ax1 = plt.subplot(3, 1, 1)
    
    # Plot curves
    line1, = ax1.plot(epochs, train_losses, 'b-', label='Train Loss', 
                      linewidth=2, alpha=0.8)
    line2, = ax1.plot(epochs, val_losses, 'r-', label='Val Loss', 
                      linewidth=2, alpha=0.8)
    
    # Mark best validation loss point
    ax1.plot(best_epoch, min_val_loss, 'g*', markersize=20, 
             label=f'Best: {min_val_loss:.6f} @ Epoch {best_epoch}', zorder=5)
    
    # Mark all saved model points
    for be in best_epochs:
        if be <= len(val_losses):
            ax1.plot(be, val_losses[be-1], 'go', markersize=8, alpha=0.5)
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ============ Subplot 2: Overfitting Analysis ============
    ax2 = plt.subplot(3, 1, 2)
    
    loss_diff = np.array(val_losses) - np.array(train_losses)
    
    ax2.plot(epochs, loss_diff, 'purple', linewidth=2.5, alpha=0.8, label='Val - Train')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Fill regions
    ax2.fill_between(epochs, 0, loss_diff, 
                     where=(np.array(loss_diff) > 0), 
                     color='red', alpha=0.2, label='Overfitting Zone')
    ax2.fill_between(epochs, 0, loss_diff, 
                     where=(np.array(loss_diff) <= 0), 
                     color='green', alpha=0.2, label='Good Generalization')
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Val Loss - Train Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ============ Subplot 3: Learning Rate Schedule ============
    if learning_rates and len(set(learning_rates)) > 1:  # Only plot if LR changes
        ax3 = plt.subplot(3, 1, 3)
        
        ax3.plot(epochs, learning_rates, 'orange', linewidth=2.5, marker='o', 
                markersize=4, alpha=0.8)
        ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')  # Log scale
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Training Loss Curves Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View VAE training curves
  python view_training_loss.py --vae
  
  # View UNet training curves  
  python view_training_loss.py --unet
  
  # View both
  python view_training_loss.py --both
  
  # Custom log file and output
  python view_training_loss.py --log-file my_training.log --output my_plot.png
        """
    )
    
    parser.add_argument('--vae', action='store_true', 
                       help='Plot VAE training curves')
    parser.add_argument('--unet', action='store_true', 
                       help='Plot UNet training curves')
    parser.add_argument('--both', action='store_true', 
                       help='Plot both VAE and UNet curves')
    parser.add_argument('--log-file', type=str, 
                       help='Specify custom log file path')
    parser.add_argument('--output', type=str, 
                       help='Specify output image path')
    parser.add_argument('--title', type=str, 
                       help='Custom chart title')
    parser.add_argument('--log-dir', type=str, 
                       default='trained_models_256Guidance/logs',
                       help='Log file directory (default: trained_models_256Guidance/logs)')
    parser.add_argument('--output-dir', type=str, 
                       default='trained_models_256Guidance/plots',
                       help='Output directory (default: trained_models_256Guidance/plots)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle custom log file
    if args.log_file:
        title = args.title or "Training Loss Curves"
        output = args.output or os.path.join(args.output_dir, 'custom_loss_curves.png')
        plot_training_curves(args.log_file, output, title)
        return
    
    # Default behavior: show help if no option specified
    if not (args.vae or args.unet or args.both):
        parser.print_help()
        print("\nTip: Use --vae, --unet or --both to view training curves")
        return
    
    # Handle preset options
    if args.both:
        args.vae = True
        args.unet = True
    
    print("\n" + "="*60)
    print("    Training Loss Curves Visualization Tool")
    print("="*60 + "\n")
    
    # Plot VAE curves
    if args.vae:
        vae_log = os.path.join(args.log_dir, 'vae_training.log')
        vae_plot = os.path.join(args.output_dir, 'vae_loss_curves.png')
        
        print("Generating VAE training curves...")
        print("-" * 60)
        plot_training_curves(vae_log, vae_plot, title="VAE Training Loss Curves")
        print()
    
    # Plot UNet curves
    if args.unet:
        unet_log = os.path.join(args.log_dir, 'unet_training.log')
        unet_plot = os.path.join(args.output_dir, 'unet_loss_curves.png')
        
        print("Generating UNet training curves...")
        print("-" * 60)
        plot_training_curves(unet_log, unet_plot, title="UNet Training Loss Curves")
        print()
    
    print("="*60)
    print(f"Done! All plots saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
