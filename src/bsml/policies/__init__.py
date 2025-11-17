
"""
Randomization policies for BSML project.

This module provides:
- Base policy class (RandomizationPolicy)
- Concrete policy implementations (Uniform, OU, Pink)
- Utility functions for seed generation and exposure checks

Owner: P4
"""

# Base class
from .base_policy import RandomizationPolicy

# Concrete policies
from .uniform_policy import (
    UniformPolicy,
    DEFAULT_UNIFORM_PARAMS,
    CONSERVATIVE_UNIFORM_PARAMS,
    AGGRESSIVE_UNIFORM_PARAMS,
    NOCLAMPING_UNIFORM_PARAMS
)

# Utility functions
from .utils import (
    generate_policy_seed,
    calculate_net_exposure,
    calculate_gross_exposure,
    is_within_exposure_tolerance,
    check_market_hours,
    clamp_to_market_hours,
    validate_parameter_bounds
)

__all__ = [
    # Base class
    'RandomizationPolicy',
    
    # Concrete policies
    'UniformPolicy',
    
    # Predefined parameter sets
    'DEFAULT_UNIFORM_PARAMS',
    'CONSERVATIVE_UNIFORM_PARAMS',
    'AGGRESSIVE_UNIFORM_PARAMS',
    'NOCLAMPING_UNIFORM_PARAMS',
    
    # Utility functions
    'generate_policy_seed',
    'calculate_net_exposure',
    'calculate_gross_exposure',
    'is_within_exposure_tolerance',
    'check_market_hours',
    'clamp_to_market_hours',
    'validate_parameter_bounds',
]

__version__ = '0.1.0'
__author__ = 'P4'