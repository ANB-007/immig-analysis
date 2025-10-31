# src/simulation/models.py
"""
Core data models for immigration simulation.
UPDATED: Added comprehensive family-adjusted backlog calculation methods.
Formula: Family backlog = (principals × SPOUSE_MULTIPLIER) + dependent_children
Now supports family-adjusted backlogs by EB category and nationality.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    TEMPORARY = "temporary"
    PERMANENT = "permanent"


class EBCategory(Enum):
    EB1 = "EB-1"
    EB2 = "EB-2"
    EB3 = "EB-3"
    EB4 = "EB-4"
    EB5 = "EB-5"


@dataclass
class Worker:
    """Individual worker with comprehensive lifecycle tracking."""
    id: int
    status: WorkerStatus
    nationality: str
    age: int
    wage: float
    created_year: int
    entry_year: int
    year_joined: int
    eb_category: Optional[EBCategory] = None
    conversion_year: Optional[int] = None
    
    @property
    def is_temporary(self) -> bool:
        return self.status == WorkerStatus.TEMPORARY
    
    @property
    def is_permanent(self) -> bool:
        return self.status == WorkerStatus.PERMANENT
    
    @property
    def was_converted(self) -> bool:
        return self.conversion_year is not None
    
    def convert_to_permanent(self, year: int):
        """Convert worker from temporary to permanent status."""
        self.status = WorkerStatus.PERMANENT
        self.conversion_year = year
        self.wage = self.wage * 1.10 # Apply wage bump on conversion
    
    def apply_wage_jump(self, factor: float):
        """Apply wage jump due to job change."""
        self.wage *= factor


@dataclass
class TemporaryWorker:
    """Lightweight representation for queue management."""
    worker_id: int
    entry_year: int
    nationality: str
    eb_category: EBCategory
    eligible_year: int = field(init=False)
    
    def __post_init__(self):
        from .empirical_params import PERM_FILING_DELAY
        self.eligible_year = self.entry_year + PERM_FILING_DELAY


@dataclass
class DependentChild:
    """Dependent child of an H-1B worker."""
    child_id: int
    parent_worker_id: int
    nationality: str
    birth_year: int
    entry_year: int
    
    def age_in_year(self, year: int) -> int:
        """Calculate child's age in given year."""
        return year - self.birth_year
    
    def years_since_entry(self, year: int) -> int:
        """Years since parent entered H-1B status."""
        return year - self.entry_year


@dataclass
class AgedOutChild:
    """Child who aged out while parent was still temporary."""
    child_id: int
    parent_worker_id: int
    nationality: str
    parent_eb_category: Optional[EBCategory]
    aged_out_year: int
    age_at_ageout: int
    parent_years_in_queue: int


@dataclass
class WageStatistics:
    avg_wage_total: float
    avg_wage_permanent: float
    avg_wage_temporary: float
    total_wage_bill: float
    
    @classmethod
    def calculate(cls, workers: List[Worker]) -> 'WageStatistics':
        if not workers:
            return cls(0.0, 0.0, 0.0, 0.0)
        
        total_wages = sum(w.wage for w in workers)
        avg_wage_total = total_wages / len(workers)
        total_wage_bill = total_wages
        
        perm_workers = [w for w in workers if w.is_permanent]
        temp_workers = [w for w in workers if w.is_temporary]
        
        avg_wage_permanent = sum(w.wage for w in perm_workers) / len(perm_workers) if perm_workers else 0.0
        avg_wage_temporary = sum(w.wage for w in temp_workers) / len(temp_workers) if temp_workers else 0.0
        
        return cls(avg_wage_total, avg_wage_permanent, avg_wage_temporary, total_wage_bill)


@dataclass
class NationalityStatistics:
    permanent_nationalities: Dict[str, int]
    temporary_nationalities: Dict[str, int]
    
    @classmethod
    def calculate(cls, workers: List[Worker]) -> 'NationalityStatistics':
        perm_nationalities = {}
        temp_nationalities = {}
        
        for worker in workers:
            if worker.is_permanent:
                perm_nationalities[worker.nationality] = perm_nationalities.get(worker.nationality, 0) + 1
            else:
                temp_nationalities[worker.nationality] = temp_nationalities.get(worker.nationality, 0) + 1
        
        return cls(perm_nationalities, temp_nationalities)
    
    def get_temporary_distribution(self) -> Dict[str, float]:
        total = sum(self.temporary_nationalities.values())
        if total == 0:
            return {}
        return {nat: count / total for nat, count in self.temporary_nationalities.items()}
    
    def get_top_temporary_nationalities(self, top_n: int = 3) -> List[Tuple[str, int]]:
        return sorted(self.temporary_nationalities.items(), key=lambda x: x, reverse=True)[:top_n]


@dataclass
class EBCategoryStatistics:
    conversions_by_category: Dict[EBCategory, int] = field(default_factory=dict)
    backlogs_by_category: Dict[EBCategory, int] = field(default_factory=dict)
    backlogs_by_category_nationality: Dict[Tuple[EBCategory, str], int] = field(default_factory=dict)
    
    @classmethod
    def calculate(cls, converted_by_category: Dict[EBCategory, int],
                  backlog_by_category: Dict[EBCategory, int],
                  backlog_by_category_nationality: Dict[Tuple[EBCategory, str], int]) -> 'EBCategoryStatistics':
        return cls(
            conversions_by_category=converted_by_category.copy(),
            backlogs_by_category=backlog_by_category.copy(),
            backlogs_by_category_nationality=backlog_by_category_nationality.copy()
        )


@dataclass
class ChildAgeoutStatistics:
    """Statistics for child age-out tracking."""
    total_aged_out: int
    aged_out_this_year: int
    children_at_risk: int
    aged_out_by_nationality: Dict[str, int]
    aged_out_by_eb_category: Dict[EBCategory, int]
    
    @classmethod
    def calculate(cls, aged_out_children: List[AgedOutChild],
                  dependent_children: List[DependentChild],
                  current_year: int,
                  aged_out_this_year: int) -> 'ChildAgeoutStatistics':
        """Calculate child age-out statistics from current data."""
        # Count aged out by nationality
        aged_out_by_nationality = defaultdict(int)
        for child in aged_out_children:
            aged_out_by_nationality[child.nationality] += 1
        
        # Count aged out by parent EB category
        aged_out_by_eb_category = defaultdict(int)
        for child in aged_out_children:
            if child.parent_eb_category:
                aged_out_by_eb_category[child.parent_eb_category] += 1
        
        # Count children currently at risk
        children_at_risk = len(dependent_children)
        
        return cls(
            total_aged_out=len(aged_out_children),
            aged_out_this_year=aged_out_this_year,
            children_at_risk=children_at_risk,
            aged_out_by_nationality=dict(aged_out_by_nationality),
            aged_out_by_eb_category=dict(aged_out_by_eb_category)
        )


@dataclass
class SimulationState:
    """Complete state of the simulation at a point in time."""
    year: int
    total_workers: int
    permanent_workers: int
    temporary_workers: int
    new_permanent: int
    new_temporary: int
    converted_temps: int
    avg_wage_total: float
    avg_wage_permanent: float
    avg_wage_temporary: float
    total_wage_bill: float
    top_temp_nationalities: List[Tuple[str, int]]
    converted_by_country: Dict[str, int]
    queue_backlog_by_country: Dict[str, int]
    country_cap_enabled: bool
    annual_conversion_cap: int
    cumulative_conversions: int
    h1b_share: float
    total_wages: float
    
    # Child age-out data
    children_aged_out_this_year: int
    cumulative_children_aged_out: int
    children_at_risk: int
    aged_out_by_nationality: Dict[str, int]
    
    # EB category data
    converted_by_eb_category: Dict[EBCategory, int]
    queue_backlog_by_eb_category: Dict[EBCategory, int]
    queue_backlog_by_eb_category_nationality: Dict[Tuple[EBCategory, str], int]
    aged_out_by_eb_category: Dict[EBCategory, int]
    
    @property
    def permanent_share(self) -> float:
        return self.permanent_workers / self.total_workers if self.total_workers > 0 else 0.0
    
    def calculate_family_adjusted_backlog(self) -> Dict[str, int]:
        """
        Calculate family-adjusted backlog: (principals × SPOUSE_MULTIPLIER) + dependent_children.
        Returns:
            Dictionary mapping nationality to family-adjusted backlog size
        """
        from .empirical_params import SPOUSE_MULTIPLIER
        family_backlog = {}
        
        for nationality, principal_count in self.queue_backlog_by_country.items():
            # Each principal counts as SPOUSE_MULTIPLIER (principal + spouse)
            # Then add the dependent children still in queue for that nationality
            family_backlog[nationality] = int(principal_count * SPOUSE_MULTIPLIER)
        
        return family_backlog
    
    def calculate_family_adjusted_backlog_with_children(self, dependent_children: List[DependentChild]) -> Dict[str, int]:
        """
        Calculate family-adjusted backlog including actual dependent children counts.
        Formula: (principals × SPOUSE_MULTIPLIER) + count of dependent children by nationality
        
        Args:
            dependent_children: List of all dependent children currently in queue
        
        Returns:
            Dictionary mapping nationality to total family backlog
        """
        from .empirical_params import SPOUSE_MULTIPLIER
        
        # Count dependent children by nationality
        children_by_nationality = defaultdict(int)
        for child in dependent_children:
            children_by_nationality[child.nationality] += 1
        
        # Calculate family backlog
        # NO GUARD - calculate for all nationalities
        family_backlog = {}
        for nationality, principal_count in self.queue_backlog_by_country.items():
            principals_and_spouses = int(principal_count * SPOUSE_MULTIPLIER)
            children_count = children_by_nationality.get(nationality, 0)
            family_backlog[nationality] = principals_and_spouses + children_count
        
        return family_backlog

    
    def get_total_family_adjusted_backlog(self, dependent_children: List[DependentChild]) -> int:
        """
        Get total family-adjusted backlog across all nationalities.
        
        Args:
            dependent_children: List of all dependent children currently in queue
        
        Returns:
            Total family backlog size
        """
        family_backlog = self.calculate_family_adjusted_backlog_with_children(dependent_children)
        return sum(family_backlog.values())


@dataclass
class BacklogAnalysis:
    """Analysis of final backlogs by nationality and EB category."""
    scenario_name: str
    total_backlog: int
    backlog_by_country: Dict[str, int]
    backlog_by_eb_category: Dict[EBCategory, int] = field(default_factory=dict)
    backlog_by_category_nationality: Dict[Tuple[EBCategory, str], int] = field(default_factory=dict)
    
    # Family-adjusted backlogs by nationality
    family_adjusted_backlog: Dict[str, int] = field(default_factory=dict)
    total_family_adjusted_backlog: int = 0
    
    # NEW: Family-adjusted backlogs by EB category
    family_adjusted_backlog_by_eb_category: Dict[EBCategory, int] = field(default_factory=dict)
    family_adjusted_backlog_by_category_nationality: Dict[Tuple[EBCategory, str], int] = field(default_factory=dict)
    
    def to_dataframe(self):
        """Convert backlog analysis to pandas DataFrame for CSV export."""
        try:
            import pandas as pd
        except ImportError:
            return {
                'scenario': self.scenario_name,
                'total_backlog': self.total_backlog,
                'total_family_adjusted_backlog': self.total_family_adjusted_backlog,
                **{f'backlog_{country}': count for country, count in self.backlog_by_country.items()},
                **{f'family_backlog_{country}': count for country, count in self.family_adjusted_backlog.items()},
                **{f'backlog_{cat.value}': count for cat, count in self.backlog_by_eb_category.items()},
                **{f'family_backlog_{cat.value}': count for cat, count in self.family_adjusted_backlog_by_eb_category.items()},
                **{f'backlog_{cat.value}_{nationality}': count 
                   for (cat, nationality), count in self.backlog_by_category_nationality.items()},
                **{f'family_backlog_{cat.value}_{nationality}': count 
                   for (cat, nationality), count in self.family_adjusted_backlog_by_category_nationality.items()}
            }
        
        rows = []
        
        # Total backlog (principals only and family-adjusted)
        rows.append({
            'scenario': self.scenario_name,
            'category': 'TOTAL',
            'nationality': 'ALL',
            'backlog_size': self.total_backlog,
            'family_adjusted_backlog': self.total_family_adjusted_backlog
        })
        
        # By country (principals only and family-adjusted)
        for country, backlog in self.backlog_by_country.items():
            family_backlog = self.family_adjusted_backlog.get(country, 0)
            rows.append({
                'scenario': self.scenario_name,
                'category': 'ALL_CATEGORIES',
                'nationality': country,
                'backlog_size': backlog,
                'family_adjusted_backlog': family_backlog
            })
        
        # By category (principals only and family-adjusted)
        for category, backlog in self.backlog_by_eb_category.items():
            family_backlog = self.family_adjusted_backlog_by_eb_category.get(category, 0)
            rows.append({
                'scenario': self.scenario_name,
                'category': category.value,
                'nationality': 'ALL',
                'backlog_size': backlog,
                'family_adjusted_backlog': family_backlog
            })
        
        # By category and nationality (principals only and family-adjusted)
        for (category, nationality), backlog in self.backlog_by_category_nationality.items():
            family_backlog = self.family_adjusted_backlog_by_category_nationality.get((category, nationality), 0)
            rows.append({
                'scenario': self.scenario_name,
                'category': category.value,
                'nationality': nationality,
                'backlog_size': backlog,
                'family_adjusted_backlog': family_backlog
            })
        
        return pd.DataFrame(rows)
    
    @classmethod
    def from_simulation(cls, sim: 'Simulation', scenario_name: str) -> 'BacklogAnalysis':
        """Create backlog analysis from completed simulation with family-adjusted calculations."""
        from .empirical_params import SPOUSE_MULTIPLIER
        
        final_state = sim.states[-1]
        total_backlog = sum(final_state.queue_backlog_by_country.values())
        
        # FILTER: Only count dependents whose parent is still temporary (still waiting)
        worker_lookup = {w.id: w for w in sim.workers}
        valid_dependent_children = []
        for child in sim.dependent_children:
            parent = worker_lookup.get(child.parent_worker_id)
            # Only include if parent exists AND is still temporary (hasn't converted)
            if parent and parent.is_temporary:
                valid_dependent_children.append(child)
        
        # Calculate family-adjusted backlog by nationality
        family_adjusted = final_state.calculate_family_adjusted_backlog_with_children(valid_dependent_children)
        total_family_adjusted = sum(family_adjusted.values())
        
        # NEW: Calculate family-adjusted backlog by EB category
        family_adjusted_by_eb = cls._calculate_family_adjusted_by_eb_category(
            final_state, valid_dependent_children, sim.workers
        )
        
        # NEW: Calculate family-adjusted backlog by (EB category, nationality)
        family_adjusted_by_cat_nat = cls._calculate_family_adjusted_by_category_nationality(
            final_state, valid_dependent_children, sim.workers
        )
        
        return cls(
            scenario_name=scenario_name,
            total_backlog=total_backlog,
            backlog_by_country=final_state.queue_backlog_by_country.copy(),
            backlog_by_eb_category=final_state.queue_backlog_by_eb_category.copy(),
            backlog_by_category_nationality=final_state.queue_backlog_by_eb_category_nationality.copy(),
            family_adjusted_backlog=family_adjusted,
            total_family_adjusted_backlog=total_family_adjusted,
            family_adjusted_backlog_by_eb_category=family_adjusted_by_eb,
            family_adjusted_backlog_by_category_nationality=family_adjusted_by_cat_nat
        )

    @staticmethod
    def _calculate_family_adjusted_by_eb_category(
        final_state: 'SimulationState',
        dependent_children: List[DependentChild],
        workers: List[Worker]
    ) -> Dict[EBCategory, int]:
        """
        Calculate family-adjusted backlog by EB category.
        Formula: (principals × SPOUSE_MULTIPLIER) + children_count
        """
        from .empirical_params import SPOUSE_MULTIPLIER
        
        # Count children by parent's EB category
        children_by_eb = defaultdict(int)
        worker_lookup = {w.id: w for w in workers}
        
        for child in dependent_children:
            parent = worker_lookup.get(child.parent_worker_id)
            if parent and parent.is_temporary and parent.eb_category:
                children_by_eb[parent.eb_category] += 1
        
        # Calculate family backlog by EB category
        # NO GUARD - calculate for all categories
        family_backlog_by_eb = {}
        for category, principal_count in final_state.queue_backlog_by_eb_category.items():
            principals_and_spouses = int(principal_count * SPOUSE_MULTIPLIER)
            children_count = children_by_eb.get(category, 0)
            family_backlog_by_eb[category] = principals_and_spouses + children_count
        
        return family_backlog_by_eb

        
    @staticmethod
    def _calculate_family_adjusted_by_category_nationality(
        final_state: 'SimulationState',
        dependent_children: List[DependentChild],
        workers: List[Worker]
    ) -> Dict[Tuple[EBCategory, str], int]:
        """
        Calculate family-adjusted backlog by (EB category, nationality).
        """
        from .empirical_params import SPOUSE_MULTIPLIER
        
        children_by_cat_nat = defaultdict(int)
        worker_lookup = {w.id: w for w in workers}
        
        for child in dependent_children:
            parent = worker_lookup.get(child.parent_worker_id)
            if parent and parent.is_temporary and parent.eb_category:
                key = (parent.eb_category, child.nationality)
                children_by_cat_nat[key] += 1
        
        # NO GUARD - calculate for all pairs, even 0 principals
        family_backlog_by_cat_nat = {}
        for (category, nationality), principal_count in final_state.queue_backlog_by_eb_category_nationality.items():
            principals_and_spouses = int(principal_count * SPOUSE_MULTIPLIER)
            children_count = children_by_cat_nat.get((category, nationality), 0)
            family_backlog_by_cat_nat[(category, nationality)] = principals_and_spouses + children_count
        
        return family_backlog_by_cat_nat


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    initial_workers: int
    years: int = 20
    seed: Optional[int] = None
    output_path: str = "data/simulation_results.csv"
    country_cap_enabled: bool = False
    compare_backlogs: bool = False
    debug: bool = False
    start_year: int = 2025
    show_nationality_summary: bool = False
    
    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return Path(self.output_path).parent
