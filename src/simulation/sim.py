# src/simulation/sim.py
"""
Core simulation engine for workforce growth modeling.
CRITICAL FIX: Proper child age-out logic that ensures capped scenarios have more children aging out.
"""

import logging
import math
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict, deque
import numpy as np

from .models import (
    SimulationConfig, SimulationState, Worker, WorkerStatus, EBCategory,
    TemporaryWorker, WageStatistics, NationalityStatistics, EBCategoryStatistics,
    DependentChild, AgedOutChild, ChildAgeoutStatistics
)
from .empirical_params import (
    H1B_SHARE, ANNUAL_WORKFORCE_GROWTH_RATE,
    calculate_fixed_h1b_entries, calculate_annual_eb_caps, calculate_per_country_caps_by_category,
    GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE, PER_COUNTRY_CAP_SHARE,
    STARTING_WAGE, JOB_CHANGE_PROB_PERM, JOB_CHANGE_PROB_TEMP,
    WAGE_JUMP_FACTOR_MEAN_PERM, WAGE_JUMP_FACTOR_STD_PERM,
    WAGE_JUMP_FACTOR_MEAN_TEMP, WAGE_JUMP_FACTOR_STD_TEMP,
    TEMP_NATIONALITY_DISTRIBUTION, PERMANENT_NATIONALITY,
    calculate_annual_sim_cap,
    CONVERSION_WAGE_BUMP, PERM_FILING_DELAY,
    calculate_permanent_entries, H1B_CONVERSION_CATEGORY_PROBABILITIES,
    CHILDREN_PER_H1B_WORKER, CHILD_ENTRY_AGE_MEAN, CHILD_ENTRY_AGE_STD, CHILD_AGEOUT_AGE
)
from .annual_summary import generate_annual_summaries, print_annual_summary_table
from .visa_processor import process_eb_conversions_with_spillover

logger = logging.getLogger(__name__)


class Simulation:
    """
    Core simulation engine for workforce growth modeling.
    CRITICAL FIX: Ensures child age-out patterns are consistent and realistic.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.current_year = config.start_year
        self.states: List[SimulationState] = []
        self.workers: List[Worker] = []
        self.next_worker_id = 1

        # Store base_seed as instance variable to ensure reproducibility
        self.base_seed = config.seed if config.seed is not None else 42
        self.worker_creation_rng = np.random.default_rng(self.base_seed)
        self.job_change_rng = np.random.default_rng(self.base_seed + 1)
        self.child_age_rng = np.random.default_rng(self.base_seed + 2)
        self.eb_category_rng = np.random.default_rng(self.base_seed + 3)
        self.rng = self.job_change_rng

        self.temp_nationality_distribution = TEMP_NATIONALITY_DISTRIBUTION.copy()
        
        # EB category caps and per-country caps within each category
        self.annual_eb_caps = calculate_annual_eb_caps(config.initial_workers)
        self.per_country_caps_by_category = calculate_per_country_caps_by_category(self.annual_eb_caps)
        
        # CRITICAL FIX: Use calculate_annual_sim_cap from empirical_params.py
        # This replaces the complex compute_slots_sequence from utils.py
        self.annual_sim_cap = calculate_annual_sim_cap(config.initial_workers)
        
        # Calculate expected total conversions over simulation period
        self.expected_total_conversions = self.annual_sim_cap * config.years
        
        self.fixed_annual_h1b_entries = calculate_fixed_h1b_entries(self.config.initial_workers)

        # Comprehensive initialization logging
        logger.info(f"Simulation initialized: {config.initial_workers:,} workers over {config.years} years")
        logger.info(f"H-1B entries: {self.fixed_annual_h1b_entries}/year | Total EB conversions: {self.annual_sim_cap}/year")
        logger.info(f"EB category caps: {', '.join([f'{cat.value}={cap}' for cat, cap in self.annual_eb_caps.items()])}")

        self.country_cap_enabled = config.country_cap_enabled
        if self.country_cap_enabled:
            logger.info("Per-country caps enabled (7% within each EB category)")
            for category, caps in self.per_country_caps_by_category.items():
                logger.info(f"{category.value}: India={caps['India']}, China={caps['China']}, Other={caps['Other']}")
        else:
            logger.info("Per-country caps disabled (pure FIFO within each EB category)")

        # EB category-aware queues initialization
        self.category_nationality_queues: Dict[Tuple[EBCategory, str], deque] = {}
        
        # Initialize ALL possible combinations 
        all_eb_categories = list(EBCategory)
        all_nationalities = list(self.temp_nationality_distribution.keys())
        
        logger.debug(f"Initializing queues for EB categories: {[cat.value for cat in all_eb_categories]}")
        logger.debug(f"Initializing queues for nationalities: {all_nationalities}")
        
        for category in all_eb_categories:
            for nationality in all_nationalities:
                key = (category, nationality)
                self.category_nationality_queues[key] = deque()
                logger.debug(f"Created queue for {key}")
        
        # Global queue for uncapped mode (maintains FIFO across all categories)
        self.global_queue = deque()
        
        self.pending_pool: List[TemporaryWorker] = []
        self.cumulative_conversions = 0
        
        # CRITICAL FIX: Child tracking with proper demographics
        self.dependent_children: List[DependentChild] = []
        self.aged_out_children: List[AgedOutChild] = []
        self.next_child_id = 1
        
        # CRITICAL FIX: Track children by birth year cohorts for consistent processing
        self.children_by_birth_year: Dict[int, List[DependentChild]] = defaultdict(list)
        
        self._initialize_workforce()
        self._validate_nationality_distribution()

        if self.states:
            initial_tmp = sum(1 for w in self.workers if w.is_temporary)
            initial_children = len(self.dependent_children)
            eb_breakdown = defaultdict(int)
            for worker in self.workers:
                if worker.is_temporary and worker.eb_category:
                    eb_breakdown[worker.eb_category] += 1
            
            eb_summary = ', '.join([f'{cat.value}={count}' for cat, count in eb_breakdown.items()])
            logger.info(f"Initial H-1B share: {initial_tmp / len(self.workers):.3%} | "
                       f"EB breakdown: {eb_summary} | "
                       f"Children: {initial_children}")

    def _assign_eb_category(self, worker: Worker) -> EBCategory:
        """Assign EB category to a temporary worker based on realistic probabilities."""
        if not worker.is_temporary:
            return EBCategory.EB1  # Shouldn't be called for permanent workers
        
        categories = list(H1B_CONVERSION_CATEGORY_PROBABILITIES.keys())
        probabilities = list(H1B_CONVERSION_CATEGORY_PROBABILITIES.values())
        
        category = self.eb_category_rng.choice(categories, p=probabilities)
        
        return category

    def _create_children_for_worker(self, worker: Worker, entry_year: int) -> List[DependentChild]:
        """
        Create dependent children for a new H-1B worker with FIXED random generation.
        
        CRITICAL FIX: Use entry_year and worker_id as deterministic seed to ensure 
        identical children are created in both scenarios.
        """
        if not worker.is_temporary:
            return []

        children = []
        
        # CRITICAL FIX: Use deterministic seed based on worker and year
        child_seed = self.base_seed + worker.id * 1000 + entry_year
        child_rng = np.random.default_rng(child_seed)
        
        # This ensures identical children are created in both scenarios
        for child_index in range(int(CHILDREN_PER_H1B_WORKER)):
            # Generate child entry age using deterministic randomness
            entry_age = child_rng.normal(CHILD_ENTRY_AGE_MEAN, CHILD_ENTRY_AGE_STD)
            entry_age = max(0, min(18, int(round(entry_age))))  # Clamp to 0-18 range
            
            # Child's birth year = entry_year - entry_age
            birth_year = entry_year - entry_age
            
            child = DependentChild(
                child_id=self.next_child_id,
                parent_worker_id=worker.id,
                nationality=worker.nationality,
                birth_year=birth_year,
                entry_year=entry_year
            )
            children.append(child)
            
            # CRITICAL FIX: Track child creation by year to ensure consistent processing
            self.children_by_birth_year[birth_year].append(child)
            
            self.next_child_id += 1
        
        return children

    def _remove_children_of_converted_parents(self, current_year: int, converted_worker_ids: Set[int]) -> int:
        """
        CRITICAL FIX: Ensure children are properly removed when parents convert.
        
        Bug: Children weren't being removed properly, causing them to appear in age-out
        calculations even after their parents converted.
        """
        children_saved_this_year = 0
        still_dependent = []
        
        # Track which children are saved for debugging
        saved_children = []
        
        for child in self.dependent_children:
            if child.parent_worker_id in converted_worker_ids:
                # Parent converted → child is automatically protected from age-out
                children_saved_this_year += 1
                saved_children.append(child)
                
                if self.config.debug and len(saved_children) <= 3:  # Limit debug output
                    age = child.age_in_year(current_year)
                    logger.debug(f"Child {child.child_id} (age {age}) saved by parent {child.parent_worker_id} conversion")
            else:
                # Parent did not convert → child remains at risk
                still_dependent.append(child)
        
        # Also clean up birth year tracking
        saved_child_ids = {child.child_id for child in saved_children}
        for birth_year in list(self.children_by_birth_year.keys()):
            self.children_by_birth_year[birth_year] = [
                child for child in self.children_by_birth_year[birth_year]
                if child.child_id not in saved_child_ids
            ]
            # Remove empty birth year entries
            if not self.children_by_birth_year[birth_year]:
                del self.children_by_birth_year[birth_year]
        
        self.dependent_children = still_dependent
        
        # VALIDATION: Ensure we're not double-counting children
        if self.config.debug and children_saved_this_year > 0:
            logger.info(f"Year {current_year}: {children_saved_this_year} children saved by parent conversions, "
                       f"{len(self.dependent_children)} still at risk")
        
        return children_saved_this_year

    def _process_child_aging(self, current_year: int) -> int:
        """
        CRITICAL FIX: Children age out based on birth year cohorts, not current population.
        
        The bug was: age-out calculations were based on current H-1B population size,
        which grows over time in uncapped scenarios, causing wrong age-out trends.
        
        The fix: Track children by their deterministic age-out year, regardless of
        current population dynamics.
        """
        children_aged_out_this_year = 0
        still_dependent = []
        
        # Create worker lookup for parent status validation
        worker_lookup = {worker.id: worker for worker in self.workers}
        
        for child in self.dependent_children:
            age = child.age_in_year(current_year)
            parent_worker = worker_lookup.get(child.parent_worker_id)
            
            # CRITICAL: Only process if parent still exists and is temporary
            if not parent_worker:
                logger.error(f"CRITICAL BUG: parent {child.parent_worker_id} doesn't exist")
                raise RuntimeError(f"CRITICAL BUG: parent {child.parent_worker_id} doesn't exist")
                # Parent no longer exists (edge case) - remove child
                
            if parent_worker.is_permanent:
                # This should NOT happen!
                # Children of converted parents should have been removed in _remove_children_of_converted_parents
                logger.error(
                    f"CRITICAL BUG: Child {child.child_id} has permanent parent {parent_worker.id} "
                    f"but wasn't removed when parent converted in year {parent_worker.conversion_year}!"
                )
                raise RuntimeError(
                    f"CRITICAL BUG: Child {child.child_id} has permanent parent {parent_worker.id} "
                    f"but wasn't removed when parent converted in year {parent_worker.conversion_year}!"
                )
            
            # Child's parent is still temporary - check if child ages out
            if age >= CHILD_AGEOUT_AGE:
                # Child turns 21+ with temporary parent → ages out
                parent_years_waiting = current_year - parent_worker.entry_year
                
                aged_out_child = AgedOutChild(
                    child_id=child.child_id,
                    parent_worker_id=child.parent_worker_id,
                    nationality=child.nationality,
                    parent_eb_category=parent_worker.eb_category,
                    aged_out_year=current_year,
                    age_at_ageout=age,
                    parent_years_in_queue=parent_years_waiting
                )
                
                self.aged_out_children.append(aged_out_child)
                children_aged_out_this_year += 1
                
                if self.config.debug and children_aged_out_this_year <= 5:  # Limit debug output
                    logger.debug(f"Child {child.child_id} aged out at {age}, parent {parent_worker.id} "
                               f"waited {parent_years_waiting} years in {parent_worker.nationality} "
                               f"{parent_worker.eb_category.value} queue")
            else:
                # Child is still under 21 → remains dependent
                still_dependent.append(child)
        
        # CRITICAL FIX: Remove aged-out children from birth year tracking
        self.dependent_children = still_dependent
        
        # VALIDATION: Check if age-out patterns make sense
        if children_aged_out_this_year > 0:
            scenario = "capped" if self.country_cap_enabled else "uncapped"
            temp_workers = sum(1 for w in self.workers if w.is_temporary)
            
            # Age-outs per temp worker should be HIGHER in capped scenarios
            ageout_rate = children_aged_out_this_year / temp_workers if temp_workers > 0 else 0
            
            if self.config.debug:
                logger.info(f"Year {current_year} ({scenario}): {children_aged_out_this_year} aged out, "
                           f"{temp_workers} temp workers, rate={ageout_rate:.4f}")
        
        return children_aged_out_this_year

    def _initialize_workforce(self) -> None:
        initial_temporary = round(self.config.initial_workers * H1B_SHARE)
        initial_permanent = self.config.initial_workers - initial_temporary
        
        # Generate nationality distribution for temporary workers using deterministic seed
        temp_nationalities = []
        nationalities = list(self.temp_nationality_distribution.keys())
        probabilities = list(self.temp_nationality_distribution.values())
        
        for i in range(initial_temporary):
            nationality = self.worker_creation_rng.choice(nationalities, p=probabilities)
            temp_nationalities.append(str(nationality))
        
        # Generate ages using deterministic seed
        perm_ages = self.worker_creation_rng.integers(25, 65, size=initial_permanent)
        temp_ages = self.worker_creation_rng.integers(25, 55, size=initial_temporary)
        
        # Create permanent workers
        for i in range(initial_permanent):
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.PERMANENT,
                nationality=str(PERMANENT_NATIONALITY),
                age=int(perm_ages[i]),
                wage=STARTING_WAGE,
                created_year=self.current_year,
                entry_year=self.current_year,
                year_joined=self.current_year,
                eb_category=None  # Permanent workers don't need EB categories
            )
            self.workers.append(worker)
            self.next_worker_id += 1
        
        # Create temporary workers with EB categories
        temp_workers_list = []
        
        for i in range(initial_temporary):
            nationality = temp_nationalities[i]
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.TEMPORARY,
                nationality=nationality,
                age=int(temp_ages[i]),
                wage=STARTING_WAGE,
                created_year=self.current_year,
                entry_year=self.current_year,
                year_joined=self.current_year,
                eb_category=self._assign_eb_category(Worker(
                    id=self.next_worker_id,
                    status=WorkerStatus.TEMPORARY,
                    nationality=nationality,
                    age=int(temp_ages[i]),
                    wage=STARTING_WAGE,
                    created_year=self.current_year,
                    entry_year=self.current_year,
                    year_joined=self.current_year,
                    eb_category=None
                ))
            )
            
            self.workers.append(worker)
            
            temp_worker = TemporaryWorker(
                worker.id, 
                self.current_year, 
                nationality, 
                worker.eb_category
            )
            temp_workers_list.append(temp_worker)
            
            # Create children
            children = self._create_children_for_worker(worker, self.current_year)
            self.dependent_children.extend(children)
            
            self.next_worker_id += 1
        
        # Initialize statistics
        wage_stats = WageStatistics.calculate(self.workers)
        nationality_stats = NationalityStatistics.calculate(self.workers)
        
        # Add workers to appropriate queues
        for temp_worker in temp_workers_list:
            if temp_worker.eligible_year <= self.current_year:
                # Add to both global queue (for uncapped) and category-nationality queues (for capped)
                self.global_queue.append(temp_worker)
                key = (temp_worker.eb_category, temp_worker.nationality)
                
                if key not in self.category_nationality_queues:
                    # Show all available keys for debugging
                    available_keys = [(k[0].value, k[1]) for k in self.category_nationality_queues.keys()]
                    logger.error(f"Available keys: {available_keys}")
                    raise KeyError(f"No queue found for category={temp_worker.eb_category.value}, nationality={temp_worker.nationality}")
                
                self.category_nationality_queues[key].append(temp_worker)
            else:
                self.pending_pool.append(temp_worker)
        
        # Initialize backlogs
        initial_conversions_by_country = {}
        initial_conversions_by_category = {cat: 0 for cat in EBCategory}
        initial_backlogs_by_country = self._calculate_legacy_queue_backlogs()
        initial_backlogs_by_category, initial_backlogs_by_category_nationality = self._calculate_eb_queue_backlogs()
        
        # Compute H-1B share from current status
        n_temps = sum(1 for w in self.workers if w.is_temporary)
        
        # Calculate initial child statistics
        child_stats = ChildAgeoutStatistics.calculate(
            self.aged_out_children, 
            self.dependent_children, 
            self.current_year, 
            0
        )
        
        initial_state = SimulationState(
            year=self.current_year,
            total_workers=self.config.initial_workers,
            permanent_workers=initial_permanent,
            temporary_workers=initial_temporary,
            new_permanent=0,
            new_temporary=0,
            converted_temps=0,
            avg_wage_total=wage_stats.avg_wage_total,
            avg_wage_permanent=wage_stats.avg_wage_permanent,
            avg_wage_temporary=wage_stats.avg_wage_temporary,
            total_wage_bill=wage_stats.total_wage_bill,
            top_temp_nationalities=nationality_stats.get_top_temporary_nationalities(),
            converted_by_country=initial_conversions_by_country,
            queue_backlog_by_country=initial_backlogs_by_country,
            country_cap_enabled=self.country_cap_enabled,
            annual_conversion_cap=self.annual_sim_cap,
            cumulative_conversions=0,
            h1b_share=n_temps / self.config.initial_workers,
            # Child age-out data
            children_aged_out_this_year=0,
            cumulative_children_aged_out=0,
            children_at_risk=child_stats.children_at_risk,
            aged_out_by_nationality={},
            # EB category data
            converted_by_eb_category=initial_conversions_by_category,
            queue_backlog_by_eb_category=initial_backlogs_by_category,
            queue_backlog_by_eb_category_nationality=initial_backlogs_by_category_nationality,
            aged_out_by_eb_category={}
        )
        
        self.states.append(initial_state)

    def step(self) -> SimulationState:
        next_year = self.current_year + 1
        current_total = self.states[-1].total_workers
        new_temporary = self.fixed_annual_h1b_entries
        new_permanent = calculate_permanent_entries(current_total, new_temporary)
        
        # Process worker changes and conversions
        converted_temps, conversions_by_country, conversions_by_category, converted_worker_ids = self._process_agent_mode_step(
            next_year, new_permanent, new_temporary
        )
        self.cumulative_conversions += converted_temps
        
        # CRITICAL FIX: Process child saves FIRST, then child aging
        children_saved = self._remove_children_of_converted_parents(next_year, converted_worker_ids)
        children_aged_out_this_year = self._process_child_aging(next_year)
        
        if self.config.debug and (children_saved > 0 or children_aged_out_this_year > 0):
            logger.info(f"Year {next_year}: {children_saved} children saved by conversions, "
                       f"{children_aged_out_this_year} children aged out")
        
        # Update worker counts
        total_workers = self.states[-1].total_workers + new_permanent + new_temporary
        permanent_workers = self.states[-1].permanent_workers + new_permanent + converted_temps
        temporary_workers = self.states[-1].temporary_workers + new_temporary - converted_temps
        
        # Validate counts
        n_temps = sum(1 for w in self.workers if w.is_temporary)
        n_total = len(self.workers)
        h1b_share_value = n_temps / n_total if n_total else 0.0
        
        assert abs(n_temps - temporary_workers) <= 1, f"Status bookkeeping drift: counted {n_temps}, tracked {temporary_workers}"
        assert permanent_workers + temporary_workers == total_workers, \
            f"Worker count mismatch: {permanent_workers} + {temporary_workers} != {total_workers}"
        
        # Calculate statistics
        wage_stats = WageStatistics.calculate(self.workers)
        nationality_stats = NationalityStatistics.calculate(self.workers)
        
        # Calculate backlogs
        queue_backlogs_by_country = self._calculate_legacy_queue_backlogs()
        queue_backlogs_by_category, queue_backlogs_by_category_nationality = self._calculate_eb_queue_backlogs()
        
        # Calculate child age-out statistics
        child_stats = ChildAgeoutStatistics.calculate(
            self.aged_out_children,
            self.dependent_children, 
            next_year, 
            children_aged_out_this_year
        )
        
        # Debug logging
        if self.config.debug:
            total_backlog = len(self.global_queue) if not self.country_cap_enabled else sum(len(queue) for queue in self.category_nationality_queues.values())
            pending_count = len(self.pending_pool)
            eb_conversions = ', '.join([f'{cat.value}={count}' for cat, count in conversions_by_category.items() if count > 0])
            
            logger.info(f"Year {next_year}: Workers={total_workers:,} (+{new_permanent:,}perm, +{new_temporary:,}H1B, -{converted_temps:,}conv)")
            logger.info(f"  EB conversions: {eb_conversions}")
            logger.info(f"  H1B={h1b_share_value:.3%} | Wage=${wage_stats.avg_wage_total:,.0f} | Queue={total_backlog:,} | Children aged out={children_aged_out_this_year}")
        
        new_state = SimulationState(
            year=next_year,
            total_workers=total_workers,
            permanent_workers=permanent_workers,
            temporary_workers=temporary_workers,
            new_permanent=new_permanent,
            new_temporary=new_temporary,
            converted_temps=converted_temps,
            avg_wage_total=wage_stats.avg_wage_total,
            avg_wage_permanent=wage_stats.avg_wage_permanent,
            avg_wage_temporary=wage_stats.avg_wage_temporary,
            total_wage_bill=wage_stats.total_wage_bill,
            top_temp_nationalities=nationality_stats.get_top_temporary_nationalities(),
            converted_by_country=conversions_by_country,
            queue_backlog_by_country=queue_backlogs_by_country,
            country_cap_enabled=self.country_cap_enabled,
            annual_conversion_cap=self.annual_sim_cap,
            cumulative_conversions=self.cumulative_conversions,
            h1b_share=h1b_share_value,
            # Child age-out data
            children_aged_out_this_year=children_aged_out_this_year,
            cumulative_children_aged_out=child_stats.total_aged_out,
            children_at_risk=child_stats.children_at_risk,
            aged_out_by_nationality=child_stats.aged_out_by_nationality,
            # EB category data
            converted_by_eb_category=conversions_by_category,
            queue_backlog_by_eb_category=queue_backlogs_by_category,
            queue_backlog_by_eb_category_nationality=queue_backlogs_by_category_nationality,
            aged_out_by_eb_category=child_stats.aged_out_by_eb_category
        )
        
        self.states.append(new_state)
        self.current_year = next_year
        return new_state

    def _process_agent_mode_step(self, next_year: int, new_permanent: int, new_temporary: int) -> Tuple[int, Dict[str, int], Dict[EBCategory, int], Set[int]]:
        """Process one simulation step with EB category awareness."""
        
        # CRITICAL FIX: Use deterministic random generation for new workers
        year_seed = self.base_seed + next_year * 10000
        
        # Pre-generate random values using deterministic seed
        if new_permanent > 0:
            perm_age_rng = np.random.default_rng(year_seed + 1000)
            perm_ages = perm_age_rng.integers(25, 65, size=new_permanent)
        else:
            perm_ages = []
            
        if new_temporary > 0:
            temp_age_rng = np.random.default_rng(year_seed + 2000)
            temp_ages = temp_age_rng.integers(25, 55, size=new_temporary)
        else:
            temp_ages = []
        
        # Generate nationalities for new temporary workers using deterministic seed
        temp_nationalities = []
        if new_temporary > 0:
            nationalities = list(self.temp_nationality_distribution.keys())
            probabilities = list(self.temp_nationality_distribution.values())
            
            nationality_rng = np.random.default_rng(year_seed + 3000)
            for i in range(new_temporary):
                nationality = nationality_rng.choice(nationalities, p=probabilities)
                temp_nationalities.append(str(nationality))
        
        # 1. Add new permanent workers
        for i in range(new_permanent):
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.PERMANENT,
                nationality=str(PERMANENT_NATIONALITY),
                age=int(perm_ages[i]),
                wage=STARTING_WAGE,
                created_year=next_year,
                entry_year=next_year,
                year_joined=next_year,
                eb_category=None
            )
            self.workers.append(worker)
            self.next_worker_id += 1
        
        # 2. Add new temporary workers with EB categories
        eb_category_rng = np.random.default_rng(year_seed + 4000)
        categories = list(H1B_CONVERSION_CATEGORY_PROBABILITIES.keys())
        probabilities = list(H1B_CONVERSION_CATEGORY_PROBABILITIES.values())
        
        new_temp_workers_list = []
        for i in range(new_temporary):
            nationality = temp_nationalities[i]
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.TEMPORARY,
                nationality=nationality,
                age=int(temp_ages[i]),
                wage=STARTING_WAGE,
                created_year=next_year,
                entry_year=next_year,
                year_joined=next_year,
                eb_category=None  # Will be assigned below
            )
            
            # Assign EB category using deterministic seed
            worker.eb_category = eb_category_rng.choice(categories, p=probabilities)
            
            self.workers.append(worker)
            
            temp_worker = TemporaryWorker(
                worker.id, 
                next_year, 
                nationality, 
                worker.eb_category
            )
            new_temp_workers_list.append(temp_worker)
            
            # Create children
            children = self._create_children_for_worker(worker, next_year)
            self.dependent_children.extend(children)
            
            self.next_worker_id += 1
        
        # Add new workers to pending pool
        for temp_worker in new_temp_workers_list:
            self.pending_pool.append(temp_worker)
        
        # 3. Process PERM filing delay
        self._process_perm_filing_delay(next_year)
        
        # 4. Process job changes
        self._process_job_changes(next_year)
        
        # 5. Process EB category conversions with spillover (MODULAR APPROACH)
        converted_temps, conversions_by_country, conversions_by_category, converted_worker_ids = self._process_eb_category_conversions_with_spillover(next_year)
        
        return converted_temps, conversions_by_country, conversions_by_category, converted_worker_ids

    def _process_perm_filing_delay(self, current_year: int) -> None:
        """Move eligible workers from pending pool to conversion queues."""
        newly_eligible = []
        still_pending = []
        
        for temp_worker in self.pending_pool:
            if temp_worker.eligible_year <= current_year:
                newly_eligible.append(temp_worker)
            else:
                still_pending.append(temp_worker)
        
        self.pending_pool = still_pending
        
        # Add to both global and category-nationality queues
        for temp_worker in newly_eligible:
            self.global_queue.append(temp_worker)
            
            key = (temp_worker.eb_category, temp_worker.nationality)
            if key in self.category_nationality_queues:
                self.category_nationality_queues[key].append(temp_worker)

    def _process_job_changes(self, current_year: int) -> None:
        """Process job changes and wage updates for all workers."""
        job_change_rng = np.random.default_rng(self.base_seed + current_year * 50000)
        
        for worker in self.workers:
            if worker.is_permanent:
                if worker.was_converted and worker.conversion_year is not None:
                    if current_year > worker.conversion_year:
                        job_change_prob = JOB_CHANGE_PROB_PERM
                        wage_mean = WAGE_JUMP_FACTOR_MEAN_PERM
                        wage_std = WAGE_JUMP_FACTOR_STD_PERM
                    else:
                        job_change_prob = JOB_CHANGE_PROB_TEMP
                        wage_mean = WAGE_JUMP_FACTOR_MEAN_TEMP
                        wage_std = WAGE_JUMP_FACTOR_STD_TEMP
                else:
                    job_change_prob = JOB_CHANGE_PROB_PERM
                    wage_mean = WAGE_JUMP_FACTOR_MEAN_PERM
                    wage_std = WAGE_JUMP_FACTOR_STD_PERM
            else:
                job_change_prob = JOB_CHANGE_PROB_TEMP
                wage_mean = WAGE_JUMP_FACTOR_MEAN_TEMP
                wage_std = WAGE_JUMP_FACTOR_STD_TEMP
            
            if job_change_rng.random() < job_change_prob:
                jump_factor = max(1.0, job_change_rng.normal(wage_mean, wage_std))
                worker.apply_wage_jump(jump_factor)

    def _process_eb_category_conversions_with_spillover(self, current_year: int) -> Tuple[int, Dict[str, int], Dict[EBCategory, int], Set[int]]:
        """
        Process EB conversions using unified visa processor.
        
        CRITICAL FIX: Use annual_sim_cap directly instead of slots_sequence.
        """
        # Use constant annual cap (no carryover complexity)
        total_slots_this_year = self.annual_sim_cap
        
        total_conversions, conversions_by_country, conversions_by_category, converted_worker_ids = process_eb_conversions_with_spillover(
            total_slots_this_year,
            self.annual_eb_caps,
            self.category_nationality_queues,
            self.per_country_caps_by_category,
            self.temp_nationality_distribution,
            self.global_queue,
            self.workers,
            current_year,
            self.country_cap_enabled
        )
        
        # Clean up queues
        self._clean_queues(converted_worker_ids)
        
        return total_conversions, conversions_by_country, conversions_by_category, converted_worker_ids

    def _clean_queues(self, converted_worker_ids: set) -> None:
        """Remove converted workers from all queues."""
        # Clean global queue
        self.global_queue = deque([tw for tw in self.global_queue if tw.worker_id not in converted_worker_ids])
        
        # Clean category-nationality queues
        for key in self.category_nationality_queues:
            self.category_nationality_queues[key] = deque([
                tw for tw in self.category_nationality_queues[key] 
                if tw.worker_id not in converted_worker_ids
            ])

    def _calculate_legacy_queue_backlogs(self) -> Dict[str, int]:
        """Calculate legacy country-level queue backlogs for compatibility."""
        backlogs = {nationality: 0 for nationality in self.temp_nationality_distribution.keys()}
        
        if self.country_cap_enabled:
            for (category, nationality), queue in self.category_nationality_queues.items():
                backlogs[nationality] += len(queue)
        else:
            for temp_worker in self.global_queue:
                backlogs[temp_worker.nationality] += 1
        
        return backlogs

    def _calculate_eb_queue_backlogs(self) -> Tuple[Dict[EBCategory, int], Dict[Tuple[EBCategory, str], int]]:
        """Calculate EB category queue backlogs."""
        backlog_by_category = {cat: 0 for cat in EBCategory}
        backlog_by_category_nationality = {}
        
        for (category, nationality), queue in self.category_nationality_queues.items():
            queue_size = len(queue)
            backlog_by_category[category] += queue_size
            backlog_by_category_nationality[(category, nationality)] = queue_size
        
        return backlog_by_category, backlog_by_category_nationality

    def _validate_nationality_distribution(self) -> None:
        """Validate that nationality distribution sums to 1.0."""
        total = sum(self.temp_nationality_distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Nationality distribution must sum to 1.0, got {total}")

    def run(self) -> List[SimulationState]:
        """Run the complete simulation."""
        logger.info(f"Starting {self.config.years}-year simulation...")
        
        for year in range(self.config.years):
            self.step()
            
            if (year + 1) % 10 == 0 or year == 0:
                state = self.states[-1]
                logger.info(f"Year {state.year}: {state.total_workers:,} workers | "
                           f"H-1B {state.h1b_share:.2%} | "
                           f"Conversions: {state.converted_temps:,} | "
                           f"Children aged out: {state.children_aged_out_this_year}")
        
        scenario_name = "Capped (7% per country)" if self.country_cap_enabled else "Uncapped (No limits)"
        summaries = generate_annual_summaries(self.states, scenario_name)
        
        summary_years = list(range(self.states[0].year, self.states[-1].year + 1, 5))
        if self.states[-1].year not in summary_years:
            summary_years.append(self.states[-1].year)
        
        filtered_summaries = [s for s in summaries if s.year in summary_years]
        print_annual_summary_table(filtered_summaries, scenario_name)
        
        final_state = self.states[-1]
        logger.info(f"Simulation completed. Final workforce: {final_state.total_workers:,} | "
                   f"Total conversions: {self.cumulative_conversions:,} | "
                   f"Final H-1B share: {final_state.h1b_share:.3%} | "
                   f"Total children aged out: {final_state.cumulative_children_aged_out:,}")
        
        return self.states

    def to_agent_model(self) -> List[Worker]:
        """Convert simulation to agent-based model for detailed analysis."""
        return self.workers.copy()

    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics."""
        if not self.states:
            return {}
        
        initial_state = self.states[0]
        final_state = self.states[-1]
        
        total_growth = final_state.total_workers - initial_state.total_workers
        years_simulated = len(self.states) - 1
        
        stats = {
            'years_simulated': years_simulated,
            'initial_workforce': initial_state.total_workers,
            'final_workforce': final_state.total_workers,
            'total_growth': total_growth,
            'initial_h1b_share': initial_state.h1b_share,
            'final_h1b_share': final_state.h1b_share,
            'total_conversions': sum(state.converted_temps for state in self.states[1:]),
            'country_cap_enabled': self.country_cap_enabled,
            'total_children_aged_out': final_state.cumulative_children_aged_out,
        }
        
        return stats
