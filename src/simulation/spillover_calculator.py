# src/simulation/spillover_calculator.py
"""
Pure mathematical spillover calculator - no artificial inefficiencies.
Models the spillover system as statutorily intended by INA Section 203(b).
"""

from typing import Dict, Tuple
from collections import deque
from .models import EBCategory


def calculate_spillover_pools(
    base_pools: Dict[EBCategory, int],
    utilization_capacities: Dict[EBCategory, int],
    realistic_spillover: bool = True
) -> Dict[EBCategory, int]:
    """
    Calculate spillover pools with mathematically perfect spillover.
    
    This models the spillover system as Congress intended:
    - Unused visas from lower-demand categories flow to higher-demand categories
    - NO visas are wasted during spillover (100% efficiency)
    - The only constraint is per-country caps within each category
    
    Args:
        base_pools: Base allocation by EB category (statutory shares)
        utilization_capacities: Maximum usable visas by category under per-country caps
        realistic_spillover: If False, applies perfect multi-iteration spillover
        
    Returns:
        Final visa pools after spillover redistribution
    """
    # Use perfect spillover for both modes
    # The "realistic" parameter is kept for API compatibility but doesn't change behavior
    return _apply_perfect_spillover(base_pools, utilization_capacities)


def _apply_perfect_spillover(
    base_pools: Dict[EBCategory, int],
    utilization_capacities: Dict[EBCategory, int]
) -> Dict[EBCategory, int]:
    """
    Apply perfect spillover as statutorily intended.
    
    Spillover cascade (INA Section 203(b)):
    1. EB-4 → EB-1 (special immigrants barely used)
    2. EB-5 → EB-1 (investors barely used)
    3. EB-1 → EB-2 (priority workers with unused visas)
    4. EB-2 → EB-3 (advanced degree with unused visas)
    5. EB-3 → EB-1 (skilled workers back to first preference if still unused)
    
    This iterates until no more spillover is possible (equilibrium reached).
    """
    pools = base_pools.copy()
    max_iterations = 10  # Prevent infinite loops (equilibrium usually reached in 2-3 iterations)
    
    for iteration in range(max_iterations):
        spillover_occurred = False
        
        # Standard spillover order per INA Section 203(b)
        spillover_pairs = [
            (EBCategory.EB4, EBCategory.EB1),  # Special immigrants → Priority workers
            (EBCategory.EB5, EBCategory.EB1),  # Investors → Priority workers
            (EBCategory.EB1, EBCategory.EB2),  # Priority workers → Advanced degree
            (EBCategory.EB2, EBCategory.EB3),  # Advanced degree → Skilled workers
            (EBCategory.EB3, EBCategory.EB1),  # Skilled workers → Priority workers (circular)
        ]
        
        for source_cat, target_cat in spillover_pairs:
            source_available = pools[source_cat]
            source_capacity = utilization_capacities.get(source_cat, 0)
            
            # How many visas can this category actually use?
            source_used = min(source_available, source_capacity)
            source_unused = source_available - source_used
            
            if source_unused > 0:
                # Transfer ALL unused visas to target category (100% efficiency)
                pools[source_cat] = source_used
                pools[target_cat] += source_unused
                spillover_occurred = True
        
        # If no spillover happened this iteration, we've reached equilibrium
        if not spillover_occurred:
            break
    
    return pools


def calculate_utilization_capacities(
    category_queues: Dict[Tuple[EBCategory, str], 'deque'],
    per_country_caps_by_category: Dict[EBCategory, Dict[str, int]],
    temp_nationality_distribution: Dict[str, float]
) -> Dict[EBCategory, int]:
    """
    Calculate maximum utilizable capacity under per-country caps.
    
    This represents the ONLY constraint that creates underutilization:
    Per-country caps within each EB category (7% rule).
    
    Example:
        EB-2 has 100 total visas available
        India queue: 500 people, but country cap = 7 visas
        China queue: 200 people, but country cap = 7 visas
        Other queue: 10 people, cap = 86 visas
        
        Utilization capacity = min(500, 7) + min(200, 7) + min(10, 86)
                             = 7 + 7 + 10 = 24 visas
        
        Unused: 100 - 24 = 76 visas spillover to EB-3
    
    Args:
        category_queues: Queue sizes by (EB category, nationality)
        per_country_caps_by_category: Per-country limits by EB category
        temp_nationality_distribution: Nationality distribution for queue keys
        
    Returns:
        Maximum usable visas by EB category (binding constraint)
    """
    utilization_capacity = {}
    
    for category in EBCategory:
        capacity = 0
        per_country_caps = per_country_caps_by_category.get(category, {})
        
        for nationality in temp_nationality_distribution.keys():
            queue_key = (category, nationality)
            queue_size = len(category_queues.get(queue_key, []))
            country_cap = per_country_caps.get(nationality, 0)
            
            # Can use at most min(queue_size, country_cap) for this nationality
            # This is the ONLY constraint - no artificial inefficiencies
            capacity += min(queue_size, country_cap)
        
        utilization_capacity[category] = capacity
    
    return utilization_capacity


def calculate_final_unused_visas(
    pools_after_spillover: Dict[EBCategory, int],
    utilization_capacities: Dict[EBCategory, int]
) -> int:
    """
    Calculate how many visas remain unused after perfect spillover.
    
    These are visas that CANNOT be used due to per-country caps binding
    across ALL categories simultaneously.
    
    Returns:
        Total unused visas (should be minimized by perfect spillover)
    """
    total_unused = 0
    
    for category, available in pools_after_spillover.items():
        capacity = utilization_capacities.get(category, 0)
        unused = max(0, available - capacity)
        total_unused += unused
    
    return total_unused
