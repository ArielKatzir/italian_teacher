"""
Reinforcement Learning module for Exercise Generation (Multi-Subject).

This module contains the reward functions and GRPO training code
for improving exercise quality through direct optimization.

Use subject-specific reward functions:
    from src.rl.reward_function.subjects.italian import ItalianRewardFunction
    from src.rl.reward_function.subjects.math import MathRewardFunction
"""

from .reward_function import ItalianRewardFunction, MathRewardFunction

__all__ = ["ItalianRewardFunction", "MathRewardFunction"]
