# src/simulation/allocation_engine.py
"""
Core visa allocation engine with realistic per-country cap enforcement.
Models real-world underutilization when caps are binding.
"""

from typing import Dict, Set, Tuple
from collections import defaultdict, deque
from .models import EBCategory, Worker


def allocate_visas_with_binding_caps(
    category_slots: int,
    category: EBCategory,
    category_nationality_queues: Dict[Tuple[EBCategory, str], deque],
    per_country_caps: Dict[str, int],
    current_year: int,
    workers_dict: Dict[int, Worker],
    conversion_wage_bump: float,
    allow_spillover_redistribution: bool = False
) -> Tuple[int, int, Dict[str, int], Set[int]]:
    """
    Allocate visas for a single EB category with BINDING per-country caps.

    Args:
        category_slots: Total visas available for this EB category this year
        category: EBCategory being processed
        category_nationality_queues: Queues keyed by (category, nationality)
        per_country_caps: Per-country cap for this category
        current_year: Simulation year
        workers_dict: Lookup of worker_id to Worker
        conversion_wage_bump: Wage multiplier on conversion
        allow_spillover_redistribution: If False, unused slots are lost

    Returns:
        slots_used: Number of visas actually allocated
        slots_unused: Number of unused visas (lost if no redistribution)
        conversions_by_country: Conversions count by nationality
        converted_worker_ids: IDs of workers converted
    """
    conversions_by_country = defaultdict(int)
    converted_worker_ids = set()
    slots_used = 0

    # Phase 1: Allocate up to per-country caps (strict enforcement)
    for nationality in sorted(per_country_caps.keys()):
        if slots_used >= category_slots:
            break

        country_cap = per_country_caps[nationality]
        queue = category_nationality_queues.get((category, nationality), deque())

        available_slots = category_slots - slots_used
        conversions_for_country = min(len(queue), country_cap, available_slots)

        for _ in range(conversions_for_country):
            temp_worker = queue.popleft()
            converted_worker_ids.add(temp_worker.worker_id)

            worker = workers_dict.get(temp_worker.worker_id)
            if worker and worker.is_temporary:
                worker.convert_to_permanent(current_year)
                worker.wage *= conversion_wage_bump

            conversions_by_country[nationality] += 1
            slots_used += 1

    # Phase 2: Handle unused slots based on spillover policy
    slots_unused = category_slots - slots_used

    if allow_spillover_redistribution and slots_unused > 0:
        remaining_slots = slots_unused
        for nationality in sorted(per_country_caps.keys()):
            if remaining_slots <= 0:
                break

            country_cap = per_country_caps[nationality]
            queue = category_nationality_queues.get((category, nationality), deque())
            already_used = conversions_by_country.get(nationality, 0)

            additional_capacity = max(0, country_cap - already_used)
            additional_conversions = min(len(queue), additional_capacity, remaining_slots)

            for _ in range(additional_conversions):
                temp_worker = queue.popleft()
                converted_worker_ids.add(temp_worker.worker_id)

                worker = workers_dict.get(temp_worker.worker_id)
                if worker and worker.is_temporary:
                    worker.convert_to_permanent(current_year)
                    worker.wage *= conversion_wage_bump

                conversions_by_country[nationality] += 1
                slots_used += 1
                remaining_slots -= 1

        slots_unused = remaining_slots

    return slots_used, slots_unused, dict(conversions_by_country), converted_worker_ids


def allocate_visas_uncapped(
    total_slots: int,
    global_queue: deque,
    current_year: int,
    workers_dict: Dict[int, Worker],
    conversion_wage_bump: float
) -> Tuple[Dict[str, int], Dict[EBCategory, int], Set[int]]:
    """
    Allocate visas without per-country caps (pure FIFO).
    Uses ALL available slots efficiently.
    """
    conversions_by_country = defaultdict(int)
    conversions_by_category = defaultdict(int)
    converted_worker_ids = set()

    slots_used = 0
    while slots_used < total_slots and global_queue:
        temp_worker = global_queue.popleft()
        converted_worker_ids.add(temp_worker.worker_id)

        worker = workers_dict.get(temp_worker.worker_id)
        if worker and worker.is_temporary:
            worker.convert_to_permanent(current_year)
            worker.wage *= conversion_wage_bump

            conversions_by_country[worker.nationality] += 1
            conversions_by_category[worker.eb_category] += 1
            slots_used += 1

    return dict(conversions_by_country), dict(conversions_by_category), converted_worker_ids
