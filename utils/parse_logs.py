"""
GRPO Training Log Parser
Extracts reward trends and training metrics from logs.txt
"""

import re
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np


def parse_logs(log_file: str = "logs.txt") -> Dict:
    """
    Parse GRPO training logs and extract key metrics.

    Returns:
        Dict with step-by-step rewards, losses, and summary statistics
    """
    with open(log_file, 'r') as f:
        content = f.read()

    # Extract reward calculation blocks
    # Pattern: "TOTAL     : min=X, max=Y, avg=Z"
    reward_pattern = r'TOTAL\s+:\s+min=[\d.]+,\s+max=[\d.]+,\s+avg=([\d.]+)'
    rewards = re.findall(reward_pattern, content)

    # Extract step numbers and losses from the training table
    # Pattern: "5	0.009000" or "10	0.015100"
    loss_pattern = r'^(\d+)\s+([\d.]+)$'
    loss_matches = re.findall(loss_pattern, content, re.MULTILINE)

    steps = [int(step) for step, _ in loss_matches]
    losses = [float(loss) for _, loss in loss_matches]

    # Extract completion counts from multi-generation verification
    # Pattern: "🔍 Step N TOTAL: X reward calls, Y completions scored"
    completion_pattern = r'🔍 Step (\d+) TOTAL: (\d+) reward calls, (\d+) completions scored'
    completion_data = re.findall(completion_pattern, content)

    # Convert to floats
    rewards = [float(r) for r in rewards]

    # Map rewards to training steps
    # Each step has 6 reward calls (from batch_size=12, grad_accum=6)
    # So group rewards by 6
    rewards_per_step = 6
    step_rewards = []
    for i in range(0, len(rewards), rewards_per_step):
        batch_rewards = rewards[i:i+rewards_per_step]
        if batch_rewards:
            step_rewards.append(sum(batch_rewards) / len(batch_rewards))

    # Build step-by-step data
    step_data = []
    max_steps = max(len(steps), len(step_rewards))

    for i in range(max_steps):
        entry = {}
        if i < len(steps):
            entry["step"] = steps[i]
            entry["loss"] = losses[i]
        if i < len(step_rewards):
            entry["avg_reward"] = step_rewards[i]
        if entry:
            step_data.append(entry)

    # Calculate statistics
    stats = {
        "total_steps": len(steps),
        "total_reward_calculations": len(rewards),
        "reward_trend": "unknown",
        "avg_reward_first_3": 0.0,
        "avg_reward_last_3": 0.0,
        "reward_change": 0.0,
        "loss_trend": "unknown",
        "avg_loss_first_3": 0.0,
        "avg_loss_last_3": 0.0,
    }

    if len(step_rewards) >= 3:
        stats["avg_reward_first_3"] = sum(step_rewards[:3]) / 3
        stats["avg_reward_last_3"] = sum(step_rewards[-3:]) / 3
        stats["reward_change"] = stats["avg_reward_last_3"] - stats["avg_reward_first_3"]

        if stats["reward_change"] > 2.0:
            stats["reward_trend"] = "improving ✅"
        elif stats["reward_change"] < -2.0:
            stats["reward_trend"] = "degrading ⚠️"
        else:
            stats["reward_trend"] = "flat/random 📊"

    if len(losses) >= 3:
        stats["avg_loss_first_3"] = sum(losses[:3]) / 3
        stats["avg_loss_last_3"] = sum(losses[-3:]) / 3
        loss_change = stats["avg_loss_last_3"] - stats["avg_loss_first_3"]

        if loss_change < -0.001:
            stats["loss_trend"] = "improving (decreasing) ✅"
        elif loss_change > 0.001:
            stats["loss_trend"] = "degrading (increasing) ⚠️"
        else:
            stats["loss_trend"] = "stable 📊"

    # Completion verification data
    completion_info = []
    for step, calls, completions in completion_data:
        completion_info.append({
            "step": int(step),
            "reward_calls": int(calls),
            "completions_scored": int(completions)
        })

    return {
        "step_data": step_data,
        "stats": stats,
        "completion_info": completion_info,
        "raw_rewards": rewards,
        "step_rewards": step_rewards,
        "raw_losses": losses,
        "steps": steps,
    }


def print_summary(data: Dict):
    """Print a clean summary of the training progress."""
    print("=" * 80)
    print("📊 GRPO TRAINING LOG ANALYSIS")
    print("=" * 80)

    stats = data["stats"]
    print(f"\n📈 TRAINING PROGRESS:")
    print(f"   Total steps logged: {stats['total_steps']}")
    print(f"   Total reward calculations: {stats['total_reward_calculations']}")

    print(f"\n🎯 REWARD ANALYSIS:")
    if data["step_rewards"]:
        print(f"   First 3 steps avg: {stats['avg_reward_first_3']:.2f}")
        print(f"   Last 3 steps avg:  {stats['avg_reward_last_3']:.2f}")
        print(f"   Change: {stats['reward_change']:+.2f}")
        print(f"   Trend: {stats['reward_trend']}")

        print(f"\n   All step-averaged rewards ({len(data['step_rewards'])} steps):")
        # Print in rows of 10 for readability
        for i in range(0, len(data['step_rewards']), 10):
            batch = data['step_rewards'][i:i+10]
            reward_strs = [f'{r:5.1f}' for r in batch]
            step_range = f"Steps {i:2d}-{min(i+9, len(data['step_rewards'])-1):2d}:"
            print(f"   {step_range:<15} {', '.join(reward_strs)}")

        print(f"\n   Total reward calculations: {len(data['raw_rewards'])} "
              f"({len(data['raw_rewards']) // len(data['step_rewards']) if data['step_rewards'] else 0} per step)")
    else:
        print("   No reward data found in logs")

    print(f"\n📉 LOSS ANALYSIS:")
    if data["raw_losses"]:
        print(f"   First 3 steps avg: {stats['avg_loss_first_3']:.4f}")
        print(f"   Last 3 steps avg:  {stats['avg_loss_last_3']:.4f}")
        print(f"   Trend: {stats['loss_trend']}")

        print(f"\n   All losses: {[f'{l:.4f}' for l in data['raw_losses'][:10]]}")
        if len(data["raw_losses"]) > 10:
            print(f"   ... (showing first 10 of {len(data['raw_losses'])})")
    else:
        print("   No loss data found in logs")

    print(f"\n🔍 MULTI-GENERATION VERIFICATION:")
    if data["completion_info"]:
        for info in data["completion_info"][:5]:
            print(f"   Step {info['step']}: {info['reward_calls']} calls, "
                  f"{info['completions_scored']} completions scored")
        if len(data["completion_info"]) > 5:
            print(f"   ... (showing first 5 of {len(data['completion_info'])})")
    else:
        print("   No completion verification data found")

    print(f"\n📋 STEP-BY-STEP DATA:")
    print(f"   {'Step':<8} {'Loss':<12} {'Avg Reward':<12}")
    print(f"   {'-'*8} {'-'*12} {'-'*12}")
    for entry in data["step_data"][:15]:
        step = entry.get("step", "N/A")
        loss = f"{entry['loss']:.4f}" if "loss" in entry else "N/A"
        reward = f"{entry['avg_reward']:.2f}" if "avg_reward" in entry else "N/A"
        print(f"   {step:<8} {loss:<12} {reward:<12}")
    if len(data["step_data"]) > 15:
        print(f"   ... (showing first 15 of {len(data['step_data'])})")

    print("\n" + "=" * 80)

    # Interpretation
    print("\n💡 INTERPRETATION:")
    if stats["reward_trend"] == "flat/random 📊":
        print("   ⚠️  Rewards are not showing improvement. Possible causes:")
        print("      - Reward signal too noisy")
        print("      - Learning rate too low/high")
        print("      - Model needs more training steps")
        print("      - Task may be too difficult for the base model")
    elif stats["reward_trend"] == "improving ✅":
        print("   ✅ Model is improving! Rewards trending upward.")
    else:
        print("   ⚠️  Model is degrading. Consider:")
        print("      - Reducing learning rate")
        print("      - Checking reward function for bugs")

    if stats["loss_trend"] == "degrading (increasing) ⚠️":
        print("\n   ⚠️  Loss is increasing - this is unusual for GRPO.")
        print("      - May indicate unstable training")
        print("      - Consider reducing learning rate")

    print("=" * 80 + "\n")


def plot_results(data: Dict, save_path: str = "training_plots.png"):
    """Create plots showing reward and loss trends."""
    step_rewards = data["step_rewards"]
    steps = data["steps"]
    losses = data["raw_losses"]

    if not step_rewards and not losses:
        print("⚠️  No data to plot")
        return

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Average Reward per Step
    if step_rewards:
        ax1 = axes[0]
        step_numbers = list(range(len(step_rewards)))
        ax1.plot(step_numbers, step_rewards, 'o-', color='#2196F3', linewidth=2,
                 markersize=6, label='Avg Reward')

        # Add trend line
        if len(step_rewards) > 1:
            z = np.polyfit(step_numbers, step_rewards, 1)
            p = np.poly1d(z)
            ax1.plot(step_numbers, p(step_numbers), "--", color='#FF5722',
                     alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2f})')

        # Add horizontal line at mean
        mean_reward = np.mean(step_rewards)
        ax1.axhline(y=mean_reward, color='gray', linestyle=':', alpha=0.5,
                    label=f'Mean={mean_reward:.1f}')

        ax1.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax1.set_title('📊 GRPO Training: Reward Progression', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Add min/max annotations
        min_idx = np.argmin(step_rewards)
        max_idx = np.argmax(step_rewards)
        ax1.annotate(f'Min: {step_rewards[min_idx]:.1f}',
                     xy=(min_idx, step_rewards[min_idx]),
                     xytext=(10, -20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        ax1.annotate(f'Max: {step_rewards[max_idx]:.1f}',
                     xy=(max_idx, step_rewards[max_idx]),
                     xytext=(10, 20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    else:
        axes[0].text(0.5, 0.5, 'No reward data available',
                     ha='center', va='center', fontsize=14)
        axes[0].set_title('📊 GRPO Training: Reward Progression', fontsize=14, fontweight='bold')

    # Plot 2: Training Loss
    if losses and steps:
        ax2 = axes[1]
        ax2.plot(steps, losses, 's-', color='#4CAF50', linewidth=2,
                 markersize=6, label='Training Loss')

        # Add trend line
        if len(losses) > 1:
            z = np.polyfit(steps, losses, 1)
            p = np.poly1d(z)
            ax2.plot(steps, p(steps), "--", color='#FF5722',
                     alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.5f})')

        ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title('📉 GRPO Training: Loss Progression', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        # Color code: green if decreasing, red if increasing
        if len(losses) >= 2:
            trend_color = 'green' if losses[-1] < losses[0] else 'red'
            ax2.spines['left'].set_color(trend_color)
            ax2.spines['left'].set_linewidth(3)
    else:
        axes[1].text(0.5, 0.5, 'No loss data available',
                     ha='center', va='center', fontsize=14)
        axes[1].set_title('📉 GRPO Training: Loss Progression', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    try:
        print("🔍 Parsing logs.txt...")
        data = parse_logs("logs.txt")
        print_summary(data)

        # Generate plots
        plot_results(data)

    except FileNotFoundError:
        print("❌ Error: logs.txt not found in current directory")
        print("   Make sure logs.txt exists in the same folder as this script")
    except Exception as e:
        print(f"❌ Error parsing logs: {e}")
        import traceback
        traceback.print_exc()
