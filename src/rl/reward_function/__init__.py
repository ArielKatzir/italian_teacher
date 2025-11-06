"""
Reward function module for multi-subject exercise generation.

Use subject-specific reward functions:
    from src.rl.reward_function.subjects.italian import ItalianRewardFunction
    from src.rl.reward_function.subjects.math import MathRewardFunction
"""

from .subjects.italian import ItalianRewardFunction
from .subjects.maths import MathRewardFunction

__all__ = [
    "ItalianRewardFunction",
    "MathRewardFunction",
]
