"""
Generate comparison plots for profit-based DQN with different activation functions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_profit_history(results_dir, experiment_name):
    """Load profit training history from JSON file."""
    history_file = results_dir / experiment_name / 'GT0_profit_history.json'
    with open(history_file) as f:
        return json.load(f)

def smooth(data, window=5):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def create_comparison_plots():
    """Create comparison plots for profit-based activation functions."""
    results_dir = Path(__file__).parent / 'results_profit_GT0'
    
    experiments = {
        'None': 'activation_none',
        'ELU': 'activation_elu',
        'Leaky ReLU': 'activation_leaky_relu',
        'PReLU': 'activation_prelu'
    }
    
    colors = {
        'None': '#1f77b4',
        'ELU': '#2ca02c',
        'Leaky ReLU': '#ff7f0e',
        'PReLU': '#d62728'
    }
    
    # Load all histories
    histories = {}
    for name, exp_dir in experiments.items():
        try:
            histories[name] = load_profit_history(results_dir, exp_dir)
            print(f"Loaded {name}: {len(histories[name]['episode'])} episodes")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Plot 1: TD Loss
    ax1 = axes[0]
    for name, hist in histories.items():
        episodes = hist['episode']
        loss = hist['loss']
        valid_idx = [i for i, l in enumerate(loss) if l is not None]
        valid_episodes = [episodes[i] for i in valid_idx]
        valid_loss = [loss[i] for i in valid_idx]
        if len(valid_loss) > 5:
            smoothed = smooth(valid_loss, window=5)
            smoothed_ep = valid_episodes[:len(smoothed)]
            ax1.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=1.5)
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('TD Loss', fontsize=11)
    ax1.set_title('TD Loss vs Episode', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Profit
    ax2 = axes[1]
    for name, hist in histories.items():
        episodes = hist['episode']
        profit = hist['profit']
        if len(profit) > 5:
            smoothed = smooth(profit, window=5)
            smoothed_ep = episodes[:len(smoothed)]
            ax2.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=1.5)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Profit ($)', fontsize=11)
    ax2.set_title('Episode Profit vs Episode', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Plot 3: Lost Demand Rate
    ax3 = axes[2]
    for name, hist in histories.items():
        episodes = hist['episode']
        lost_rate = hist['lost_demand_rate']
        if len(lost_rate) > 5:
            smoothed = smooth(lost_rate, window=5)
            smoothed_ep = episodes[:len(smoothed)]
            ax3.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=1.5)
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Lost Demand Rate (%)', fontsize=11)
    ax3.set_title('Lost Demand Rate vs Episode', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save combined plot
    output_path = results_dir / 'profit_activation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Combined plot saved to: {output_path}")
    
    # Save individual high-res plots
    for i, (title, ylabel, key) in enumerate([
        ('TD Loss', 'TD Loss', 'loss'),
        ('Profit', 'Profit ($)', 'profit'),
        ('Lost Demand Rate', 'Lost Demand Rate (%)', 'lost_demand_rate')
    ]):
        fig_single, ax_single = plt.subplots(figsize=(8, 5))
        for name, hist in histories.items():
            episodes = hist['episode']
            data = hist[key]
            if key == 'loss':
                valid_idx = [j for j, l in enumerate(data) if l is not None]
                valid_episodes = [episodes[j] for j in valid_idx]
                valid_data = [data[j] for j in valid_idx]
                if len(valid_data) > 5:
                    smoothed = smooth(valid_data, window=5)
                    smoothed_ep = valid_episodes[:len(smoothed)]
                    ax_single.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=2)
                ax_single.set_yscale('log')
            else:
                if len(data) > 5:
                    smoothed = smooth(data, window=5)
                    smoothed_ep = episodes[:len(smoothed)]
                    ax_single.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=2)
                if key == 'profit':
                    ax_single.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, label='Break-even')
        
        ax_single.set_xlabel('Episode', fontsize=12)
        ax_single.set_ylabel(ylabel, fontsize=12)
        ax_single.set_title(f'{title} by Output Activation (Profit-DQN, 50K steps)', fontsize=13, fontweight='bold')
        ax_single.legend(loc='best', fontsize=10)
        ax_single.grid(True, alpha=0.3)
        
        single_path = results_dir / f'profit_{title.lower().replace(" ", "_")}.png'
        fig_single.savefig(single_path, dpi=150, bbox_inches='tight')
        print(f"✅ {title} plot saved to: {single_path}")
        plt.close(fig_single)
    
    plt.close(fig)
    
    # Print summary
    print("\n" + "="*60)
    print("PROFIT-DQN TRAINING SUMMARY (50K steps)")
    print("="*60)
    
    for name, hist in histories.items():
        final_profit = hist['profit'][-1] if hist['profit'] else 0
        final_lost = hist['lost_demand_rate'][-1] if hist['lost_demand_rate'] else 0
        max_profit = max(hist['profit']) if hist['profit'] else 0
        print(f"{name:15} -> Final Profit: ${final_profit:.2f}, Max: ${max_profit:.2f}, Lost: {final_lost:.1f}%")
    
    return output_path

if __name__ == "__main__":
    create_comparison_plots()
