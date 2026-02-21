"""
Generate comparison plots for different activation functions.
Compares TD loss, episode return, and lost demand across training steps.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_history(results_dir, experiment_name):
    """Load training history from JSON file."""
    history_file = results_dir / experiment_name / 'GT0_training_history.json'
    with open(history_file) as f:
        return json.load(f)

def smooth(data, window=5):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def create_comparison_plots():
    """Create comparison plots for activation functions."""
    results_dir = Path(__file__).parent / 'results_GT0'
    
    # Activation function experiments
    experiments = {
        'None': 'activation_none',
        'ELU': 'activation_elu',
        'Leaky ReLU': 'activation_leaky_relu',
        'PReLU': 'activation_prelu'
    }
    
    # Colors for each activation
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
            histories[name] = load_training_history(results_dir, exp_dir)
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
        # Filter out None values
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
    
    # Plot 2: Episode Return (Total Reward)
    ax2 = axes[1]
    for name, hist in histories.items():
        episodes = hist['episode']
        reward = hist['total_reward']
        if len(reward) > 5:
            smoothed = smooth(reward, window=5)
            smoothed_ep = episodes[:len(smoothed)]
            ax2.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=1.5)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Episode Return', fontsize=11)
    ax2.set_title('Episode Return vs Episode', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
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
    ax3.axhline(y=13.49, color='gray', linestyle='--', linewidth=1, label='Static Baseline')
    
    plt.tight_layout()
    
    # Save plot
    output_path = results_dir / 'activation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {output_path}")
    
    # Also save individual high-res versions
    for i, (title, ylabel) in enumerate([
        ('TD Loss', 'TD Loss'),
        ('Episode Return', 'Episode Return'),
        ('Lost Demand Rate', 'Lost Demand Rate (%)')
    ]):
        fig_single, ax_single = plt.subplots(figsize=(8, 5))
        for name, hist in histories.items():
            episodes = hist['episode']
            if i == 0:
                data = hist['loss']
                valid_idx = [j for j, l in enumerate(data) if l is not None]
                valid_episodes = [episodes[j] for j in valid_idx]
                valid_data = [data[j] for j in valid_idx]
                if len(valid_data) > 5:
                    smoothed = smooth(valid_data, window=5)
                    smoothed_ep = valid_episodes[:len(smoothed)]
                    ax_single.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=2)
                ax_single.set_yscale('log')
            elif i == 1:
                data = hist['total_reward']
                if len(data) > 5:
                    smoothed = smooth(data, window=5)
                    smoothed_ep = episodes[:len(smoothed)]
                    ax_single.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=2)
            else:
                data = hist['lost_demand_rate']
                if len(data) > 5:
                    smoothed = smooth(data, window=5)
                    smoothed_ep = episodes[:len(smoothed)]
                    ax_single.plot(smoothed_ep, smoothed, label=name, color=colors[name], linewidth=2)
                ax_single.axhline(y=13.49, color='gray', linestyle='--', linewidth=1.5, label='Static Baseline (13.49%)')
        
        ax_single.set_xlabel('Episode', fontsize=12)
        ax_single.set_ylabel(ylabel, fontsize=12)
        ax_single.set_title(f'{title} by Output Activation Function (20K steps)', fontsize=13, fontweight='bold')
        ax_single.legend(loc='best', fontsize=10)
        ax_single.grid(True, alpha=0.3)
        
        single_path = results_dir / f'activation_{title.lower().replace(" ", "_")}.png'
        fig_single.savefig(single_path, dpi=150, bbox_inches='tight')
        print(f"✅ {title} plot saved to: {single_path}")
        plt.close(fig_single)
    
    plt.close(fig)
    
    # Print final results summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY (20K steps, 50 test episodes)")
    print("="*60)
    
    # Load evaluation results
    for name, exp_dir in experiments.items():
        eval_file = results_dir / exp_dir / 'evaluation' / 'GT0_evaluation_results.json'
        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
            print(f"{name:15} -> {eval_data['mean_lost_demand_rate']:.2f}% ± {eval_data['std_lost_demand_rate']:.2f}%")
        except:
            pass
    
    return output_path

if __name__ == "__main__":
    create_comparison_plots()
