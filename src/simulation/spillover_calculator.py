# src/simulation/spillover_calculator.py
"""
Spillover Allocation Calculator
Handles redistribution of unused EB slots from low-demand to high-demand countries.

COMPLETELY REWRITTEN: Removed ALL per-country cap constraints from spillover phase.
Spillover is now allocated proportionally to backlog size ONLY, allowing countries
to exceed their initial 7% allocation through spillover.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def calculate_spillover_allocations(
    available_slots: int,
    per_country_caps: Dict[str, int],
    backlog_by_country: Dict[str, int],
    used_by_country: Dict[str, int]
) -> Dict[str, int]:
    """
    Calculate spillover slot allocations for countries with remaining demand.
    
    Spillover rules (COMPLETELY REWRITTEN):
    1. Slots come from countries that didn't use their full allocation
    2. Distributed to countries with backlogs PROPORTIONALLY to backlog size
    3. NO per-country cap constraint on spillover - countries can exceed 7%
    4. Allocation based ONLY on backlog size, not capacity remaining
    
    Args:
        available_slots: Total unused slots available for spillover
        per_country_caps: Original per-country cap for each country (NOT USED IN SPILLOVER)
        backlog_by_country: Current backlog size for each country
        used_by_country: Not used in spillover phase
        
    Returns:
        Dict mapping country -> additional slots allocated through spillover
    """
    if available_slots <= 0:
        return {country: 0 for country in per_country_caps.keys()}
    
    spillover_allocations = {country: 0 for country in per_country_caps.keys()}
    
    # Step 1: Get total backlog across ALL countries (no capacity checks)
    total_backlog = sum(backlog_by_country.get(country, 0) for country in per_country_caps.keys())
    
    if total_backlog == 0:
        logger.debug("No backlog for spillover distribution")
        return spillover_allocations
    
    # Step 2: Allocate spillover proportionally to each country's backlog
    # NO per-country cap checks - only backlog matters
    slots_allocated = 0
    countries = list(per_country_caps.keys())
    
    for i, country in enumerate(countries):
        backlog = backlog_by_country.get(country, 0)
        
        if backlog == 0:
            spillover_allocations[country] = 0
            continue
        
        # For last country, give all remaining slots (handles rounding)
        if i == len(countries) - 1:
            allocation = available_slots - slots_allocated
        else:
            # Allocate proportionally: (country_backlog / total_backlog) * available_slots
            proportion = backlog / total_backlog
            allocation = int(proportion * available_slots)
        
        spillover_allocations[country] = allocation
        slots_allocated += allocation
        
        logger.debug(f"Spillover: {country} gets {allocation} slots "
                    f"(backlog={backlog}, proportion={backlog/total_backlog:.2%})")
    
    return spillover_allocations
