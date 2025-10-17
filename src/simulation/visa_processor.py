"""
Unified EB Visa Processing Engine
Handles all employment-based green card conversions with per-country caps and spillover.

This module consolidates visa processing logic previously split between allocation_engine.py
and visa_processor.py, delegating spillover calculations to spillover_calculator.py.
"""

import logging
from typing import List, Dict, Tuple, Set, Deque
from collections import defaultdict, deque

from .models import Worker, TemporaryWorker, EBCategory
from .spillover_calculator import calculate_spillover_allocations

logger = logging.getLogger(__name__)


class VisaProcessor:
    """
    Unified visa processor for EB green card conversions.

    Handles both capped (per-country limits) and uncapped (pure FIFO) scenarios.
    Delegates spillover logic to spillover_calculator module.
    """

    def __init__(
        self,
        country_cap_enabled: bool = False,
        per_country_cap_share: float = 0.07
    ):
        """
        Initialize visa processor.

        Args:
            country_cap_enabled: Whether to apply 7% per-country caps
            per_country_cap_share: Share allocated to each country (default 0.07 = 7%)
        """
        self.country_cap_enabled = country_cap_enabled
        self.per_country_cap_share = per_country_cap_share

        # EB category allocation shares (28.6% each for EB1-3)
        self.eb_category_shares = {
            EBCategory.EB1: 0.286,
            EBCategory.EB2: 0.286,
            EBCategory.EB3: 0.286,
            EBCategory.EB4: 0.071,
            EBCategory.EB5: 0.071
        }

        logger.info(f"VisaProcessor initialized: country_cap={'enabled' if country_cap_enabled else 'disabled'}")

    def process_conversions(
        self,
        total_slots: int,
        annual_eb_caps: Dict[EBCategory, int],
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[TemporaryWorker]],
        per_country_caps_by_category: Dict[EBCategory, Dict[str, int]],
        nationality_distribution: Dict[str, float],
        global_queue: Deque[TemporaryWorker],
        workers: List[Worker],
        current_year: int
    ) -> Tuple[int, Dict[str, int], Dict[EBCategory, int], Set[int]]:
        """
        Process EB conversions for current year.

        This is the main entry point that routes to capped or uncapped processing.

        Returns:
            total_conversions: Number of workers converted
            conversions_by_country: Dict of conversions per nationality
            conversions_by_category: Dict of conversions per EB category
            converted_worker_ids: Set of converted worker IDs
        """
        if self.country_cap_enabled:
            return self._process_capped_conversions(
                total_slots,
                annual_eb_caps,
                category_nationality_queues,
                per_country_caps_by_category,
                workers,
                current_year
            )
        else:
            return self._process_uncapped_conversions(
                total_slots,
                annual_eb_caps,
                global_queue,
                workers,
                current_year
            )

    def _process_uncapped_conversions(
        self,
        total_slots: int,
        annual_eb_caps: Dict[EBCategory, int],
        global_queue: Deque[TemporaryWorker],
        workers: List[Worker],
        current_year: int
    ) -> Tuple[int, Dict[str, int], Dict[EBCategory, int], Set[int]]:
        """
        Process conversions without per-country caps (pure FIFO across all categories).
        """
        converted_worker_ids = set()
        conversions_by_country = defaultdict(int)
        conversions_by_category = {cat: 0 for cat in EBCategory}
        slots_used_by_category = {cat: 0 for cat in EBCategory}

        worker_lookup = {w.id: w for w in workers}
        processed_temp_workers = []

        while global_queue and sum(slots_used_by_category.values()) < total_slots:
            temp_worker = global_queue.popleft()
            category = temp_worker.eb_category
            category_limit = annual_eb_caps.get(category, 0)

            if slots_used_by_category[category] >= category_limit:
                processed_temp_workers.append(temp_worker)
                continue

            worker = worker_lookup.get(temp_worker.worker_id)
            if not worker or not worker.is_temporary:
                continue

            worker.convert_to_permanent(current_year)
            converted_worker_ids.add(worker.id)
            conversions_by_country[worker.nationality] += 1
            conversions_by_category[category] += 1
            slots_used_by_category[category] += 1

        for temp_worker in processed_temp_workers:
            global_queue.appendleft(temp_worker)

        total_conversions = len(converted_worker_ids)
        logger.info(f"Year {current_year} (Uncapped): Converted {total_conversions} workers")
        logger.debug(f"  By category: {dict(conversions_by_category)}")
        return total_conversions, dict(conversions_by_country), conversions_by_category, converted_worker_ids

    def _process_capped_conversions(
        self,
        total_slots: int,
        annual_eb_caps: Dict[EBCategory, int],
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[TemporaryWorker]],
        per_country_caps_by_category: Dict[EBCategory, Dict[str, int]],
        workers: List[Worker],
        current_year: int
    ) -> Tuple[int, Dict[str, int], Dict[EBCategory, int], Set[int]]:
        """
        Process conversions WITH per-country caps (7% limit per country per category).
        """
        converted_worker_ids = set()
        conversions_by_country = defaultdict(int)
        conversions_by_category = {cat: 0 for cat in EBCategory}

        worker_lookup = {w.id: w for w in workers}

        for eb_category in [EBCategory.EB1, EBCategory.EB2, EBCategory.EB3]:
            category_slots = annual_eb_caps.get(eb_category, 0)
            if category_slots == 0:
                continue

            phase1_conversions, phase1_by_nat = self._apply_per_country_caps(
                eb_category,
                category_slots,
                per_country_caps_by_category[eb_category],
                category_nationality_queues,
                worker_lookup,
                current_year,
                converted_worker_ids,
                conversions_by_country,
                conversions_by_category
            )

            slots_remaining = category_slots - phase1_conversions
            if slots_remaining > 0:
                spillover_conversions = self._apply_spillover(
                    eb_category,
                    slots_remaining,
                    per_country_caps_by_category[eb_category],
                    category_nationality_queues,
                    worker_lookup,
                    current_year,
                    converted_worker_ids,
                    conversions_by_country,
                    conversions_by_category,
                    phase1_by_nat
                )
                logger.debug(f"  {eb_category.value} spillover: {spillover_conversions} additional conversions")

        total_conversions = len(converted_worker_ids)
        logger.info(f"Year {current_year} (Capped): Converted {total_conversions} workers")
        logger.debug(f"  By category: {dict(conversions_by_category)}")
        logger.debug(f"  By country: {dict(conversions_by_country)}")
        return total_conversions, dict(conversions_by_country), conversions_by_category, converted_worker_ids

    def _apply_per_country_caps(
        self,
        eb_category: EBCategory,
        category_slots: int,
        per_country_caps: Dict[str, int],
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[TemporaryWorker]],
        worker_lookup: Dict[int, Worker],
        current_year: int,
        converted_worker_ids: Set[int],
        conversions_by_country: Dict[str, int],
        conversions_by_category: Dict[EBCategory, int]
    ) -> Tuple[int, Dict[str, int]]:
        """
        Apply per-country caps strictly (Phase 1).
        Returns total conversions and conversions by nationality.
        """
        phase1_conversions = 0
        phase1_by_nationality = defaultdict(int)

        for nationality, country_cap in per_country_caps.items():
            queue_key = (eb_category, nationality)
            if queue_key not in category_nationality_queues:
                continue
            queue = category_nationality_queues[queue_key]
            conversions_made = 0

            while queue and conversions_made < country_cap:
                temp_worker = queue.popleft()
                worker = worker_lookup.get(temp_worker.worker_id)
                if not worker or not worker.is_temporary:
                    continue
                worker.convert_to_permanent(current_year)
                converted_worker_ids.add(worker.id)
                conversions_by_country[nationality] += 1
                conversions_by_category[eb_category] += 1
                conversions_made += 1
                phase1_conversions += 1
                phase1_by_nationality[nationality] += 1

            logger.debug(f"  {eb_category.value}-{nationality}: {conversions_made}/{country_cap} (Phase 1)")

        return phase1_conversions, dict(phase1_by_nationality)

    def _apply_spillover(
        self,
        eb_category: EBCategory,
        slots_remaining: int,
        per_country_caps: Dict[str, int],
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[TemporaryWorker]],
        worker_lookup: Dict[int, Worker],
        current_year: int,
        converted_worker_ids: Set[int],
        conversions_by_country: Dict[str, int],
        conversions_by_category: Dict[EBCategory, int],
        phase1_by_nationality: Dict[str, int]
    ) -> int:
        """
        Apply spillover (Phase 2), using per-category phase1_by_nationality.
        """
        backlog_by_country = {}
        used_by_country = {}

        for nationality in per_country_caps.keys():
            queue_key = (eb_category, nationality)
            backlog_by_country[nationality] = len(category_nationality_queues.get(queue_key, []))
            used_by_country[nationality] = phase1_by_nationality.get(nationality, 0)

        spillover_allocations = calculate_spillover_allocations(
            slots_remaining,
            per_country_caps,
            backlog_by_country,
            used_by_country
        )

        spillover_conversions = 0
        for nationality, additional_slots in spillover_allocations.items():
            if additional_slots == 0:
                continue
            queue_key = (eb_category, nationality)
            queue = category_nationality_queues.get(queue_key, deque())
            conversions_made = 0

            while queue and conversions_made < additional_slots:
                temp_worker = queue.popleft()
                worker = worker_lookup.get(temp_worker.worker_id)
                if not worker or not worker.is_temporary:
                    continue
                worker.convert_to_permanent(current_year)
                converted_worker_ids.add(worker.id)
                conversions_by_country[nationality] += 1
                conversions_by_category[eb_category] += 1
                conversions_made += 1
                spillover_conversions += 1

            logger.debug(f"  {eb_category.value}-{nationality}: +{conversions_made} (Spillover)")

        return spillover_conversions


def process_eb_conversions_with_spillover(
    total_slots: int,
    annual_eb_caps: Dict[EBCategory, int],
    category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[TemporaryWorker]],
    per_country_caps_by_category: Dict[EBCategory, Dict[str, int]],
    nationality_distribution: Dict[str, float],
    global_queue: Deque[TemporaryWorker],
    workers: List[Worker],
    current_year: int,
    country_cap_enabled: bool
) -> Tuple[int, Dict[str, int], Dict[EBCategory, int], Set[int]]:
    """
    Convenience function for backward compatibility with existing sim.py code.
    """
    processor = VisaProcessor(country_cap_enabled=country_cap_enabled)
    return processor.process_conversions(
        total_slots=total_slots,
        annual_eb_caps=annual_eb_caps,
        category_nationality_queues=category_nationality_queues,
        per_country_caps_by_category=per_country_caps_by_category,
        nationality_distribution=nationality_distribution,
        global_queue=global_queue,
        workers=workers,
        current_year=current_year
    )
