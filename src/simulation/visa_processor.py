# src/simulation/visa_processor.py
"""
Visa processor with EB category-aware allocation and spillover.
FIXED: Properly passes category-nationality queues to allocation engine.
"""

from typing import Dict, Set, Tuple
from collections import defaultdict, deque
from .models import EBCategory
from .allocation_engine import allocate_visas_with_binding_caps, allocate_visas_uncapped
from .spillover_calculator import calculate_spillover_pools, calculate_utilization_capacities


def process_eb_conversions_with_spillover(
    total_slots: int,
    annual_eb_caps: Dict[EBCategory, int],
    category_nationality_queues: Dict[Tuple[EBCategory, str], deque],
    per_country_caps_by_category: Dict[EBCategory, Dict[str, int]],
    temp_nationality_distribution: Dict[str, float],
    global_queue: deque,
    workers: list,
    current_year: int,
    country_cap_enabled: bool
) -> Tuple[int, Dict[str, int], Dict[EBCategory, int], Set[int]]:
    """
    Process EB conversions with category-aware spillover.
    
    FIXED: Passes the full category_nationality_queues dict to allocation engine,
    which extracts the correct (category, nationality) tuples.
    """
    
    # Step 1: Calculate base pools from annual caps
    base_pools = {}
    total_eb_cap = sum(annual_eb_caps.values())
    
    for category, annual_cap in annual_eb_caps.items():
        if total_eb_cap > 0:
            base_pools[category] = round(total_slots * (annual_cap / total_eb_cap))
        else:
            base_pools[category] = 0
    
    # Adjust for rounding
    actual_total = sum(base_pools.values())
    if actual_total != total_slots:
        base_pools[EBCategory.EB2] += (total_slots - actual_total)
    
    # Create worker lookup for efficiency
    workers_dict = {worker.id: worker for worker in workers}
    
    conversions_by_country = defaultdict(int)
    conversions_by_category = {cat: 0 for cat in EBCategory}
    converted_worker_ids = set()
    
    if country_cap_enabled:
        # CAPPED MODE: Process with per-country caps and spillover
        
        # Step 2: Calculate utilization capacities (binding constraint from per-country caps)
        utilization_capacities = calculate_utilization_capacities(
            category_nationality_queues,
            per_country_caps_by_category,
            temp_nationality_distribution
        )
        
        # Step 3: Apply spillover (perfect spillover, no artificial losses)
        spillover_pools = calculate_spillover_pools(
            base_pools,
            utilization_capacities,
            realistic_spillover=True
        )
        
        # Step 4: Process each category with binding per-country caps
        for category in [EBCategory.EB1, EBCategory.EB2, EBCategory.EB3, EBCategory.EB4, EBCategory.EB5]:
            category_slots = spillover_pools.get(category, 0)
            if category_slots <= 0:
                continue
            
            per_country_caps = per_country_caps_by_category.get(category, {})
            
            # CRITICAL FIX: Pass the FULL category_nationality_queues dict
            # The allocation engine will extract (category, nationality) tuples
            used_slots, unused_slots, conv_by_country, conv_ids = allocate_visas_with_binding_caps(
                category_slots=category_slots,
                category=category,
                category_nationality_queues=category_nationality_queues,  # Pass the full dict
                per_country_caps=per_country_caps,
                current_year=current_year,
                workers_dict=workers_dict,
                conversion_wage_bump=1.06,
                allow_spillover_redistribution=False
            )
            
            # Aggregate results
            conversions_by_category[category] = used_slots
            converted_worker_ids.update(conv_ids)
            
            for nationality, count in conv_by_country.items():
                conversions_by_country[nationality] += count
        
        total_conversions = sum(conversions_by_category.values())
        
    else:
        # UNCAPPED MODE: Pure FIFO, no per-country caps
        conv_country, conv_category, conv_ids = allocate_visas_uncapped(
            total_slots, global_queue, current_year, workers_dict, 1.06
        )
        conversions_by_country.update(conv_country)
        conversions_by_category.update(conv_category)
        converted_worker_ids.update(conv_ids)
        total_conversions = sum(conversions_by_category.values())
    
    return total_conversions, dict(conversions_by_country), conversions_by_category, converted_worker_ids
