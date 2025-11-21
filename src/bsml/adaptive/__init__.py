"""
P7 Adaptive Adversary Module

Provides adaptive training loop that dynamically adjusts policy parameters
based on adversary predictability (AUC scores).

Owner: P7
Week: 3
"""

from .adaptive_loop_v1 import adaptive_training_loop, main

__all__ = ['adaptive_training_loop', 'main']
__version__ = '1.0.0'
__author__ = 'P7'
