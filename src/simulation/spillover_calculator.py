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
    Fully distributes 'available_slots' proportionally to backlog.
    """
    if available_slots <= 0:
        return {country: 0 for country in per_country_caps.keys()}

    total_backlog = sum(backlog_by_country.get(country, 0) for country in per_country_caps.keys())
    if total_backlog == 0:
        logger.debug("No backlog for spillover distribution")
        return {country: 0 for country in per_country_caps.keys()}

    slots_allocated = 0
    countries = list(per_country_caps.keys())
    floored_allocations = {}
    for country in countries:
        backlog = backlog_by_country.get(country, 0)
        if backlog == 0:
            floored_allocations[country] = 0
        else:
            proportion = backlog / total_backlog
            allocation = int(proportion * available_slots)
            floored_allocations[country] = allocation
            slots_allocated += allocation

    remainder = available_slots - slots_allocated
    sorted_by_backlog = sorted(countries, key=lambda c: backlog_by_country.get(c, 0), reverse=True)
    for country in sorted_by_backlog:
        if remainder <= 0:
            break
        if backlog_by_country.get(country, 0) == 0:
            continue
        floored_allocations[country] += 1
        remainder -= 1

    return floored_allocations
