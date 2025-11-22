"""
P7 Adaptive Adversary Module

Predicts: Will a trade occur in the next period?
(Daily prediction task, but with adaptive parameter tuning)
"""

from .adaptive_loop_v1 import main, adaptive_training_loop

__all__ = ['main', 'adaptive_training_loop']
