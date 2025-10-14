"""
Reinforcement Learning module for Italian Exercise Generation.

This module contains the reward function and GRPO training code
for improving exercise quality through direct optimization.
"""

from .reward_function import ExerciseRewardFunction, score_exercise

__all__ = ["ExerciseRewardFunction", "score_exercise"]
