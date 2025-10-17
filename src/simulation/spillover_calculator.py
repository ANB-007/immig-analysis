"""
Spillover Allocation Calculator
Handles redistribution of unused EB slots from low-demand to high-demand countries.
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
    
    Spillover rules:
    1. Slots come from countries that didn't use their full allocation
    2. Distributed to countries with backlogs, prioritizing highest backlogs
    3. Still respects per-country cap (can't exceed original cap even with spillover)
    
    Args:
        available_slots: Total unused slots available for spillover
        per_country_caps: Original per-country cap for each country
        backlog_by_country: Current backlog size for each country
        used_by_country: How many slots each country already used in Phase 1
        
    Returns:
        Dict mapping country -> additional slots allocated through spillover
    """
    if available_slots <= 0:
        return {country: 0 for country in per_country_caps.keys()}
    
    spillover_allocations = {country: 0 for country in per_country_caps.keys()}
    
    # Identify countries with remaining capacity
    countries_with_demand = []
    
    for country in per_country_caps.keys():
        backlog = backlog_by_country.get(country, 0)
        used = used_by_country.get(country, 0)
        cap = per_country_caps[country]
        
        # Country has demand if: has backlog AND hasn't hit per-country cap yet
        remaining_capacity = cap - used
        
        if backlog > 0 and remaining_capacity > 0:
            countries_with_demand.append({
                'country': country,
                'backlog': backlog,
                'remaining_capacity': remaining_capacity
            })
    
    if not countries_with_demand:
        logger.debug("No countries with remaining demand for spillover")
        return spillover_allocations
    
    # Sort by backlog size (highest first)
    countries_with_demand.sort(key=lambda x: x['backlog'], reverse=True)
    
    # Distribute spillover slots
    slots_to_distribute = available_slots
    
    for country_info in countries_with_demand:
        if slots_to_distribute <= 0:
            break
        
        country = country_info['country']
        remaining_capacity = country_info['remaining_capacity']
        backlog = country_info['backlog']
        
        # Allocate up to remaining capacity or available slots, whichever is smaller
        allocation = min(remaining_capacity, backlog, slots_to_distribute)
        
        spillover_allocations[country] = allocation
        slots_to_distribute -= allocation
        
        logger.debug(f"Spillover: {country} gets +{allocation} slots "
                    f"(backlog={backlog}, capacity={remaining_capacity})")
    
    return spillover_allocations
