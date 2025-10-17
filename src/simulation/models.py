# src/simulation/models.py
"""
Core data models for immigration simulation.
COMPLETE FIX: All required models for child age-out tracking.
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
        return sorted(self.temporary_nationalities.items(), key=lambda x: x[1], reverse=True)[:top_n]

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

@dataclass
class BacklogAnalysis:
    """Analysis of final backlogs by nationality and EB category."""
    scenario_name: str
    total_backlog: int
    backlog_by_country: Dict[str, int]
    backlog_by_eb_category: Dict[EBCategory, int] = field(default_factory=dict)
    backlog_by_category_nationality: Dict[Tuple[EBCategory, str], int] = field(default_factory=dict)
    
    def to_dataframe(self):
        """Convert backlog analysis to pandas DataFrame for CSV export."""
        try:
            import pandas as pd
        except ImportError:
            return {
                'scenario': self.scenario_name,
                'total_backlog': self.total_backlog,
                **{f'backlog_{country}': count for country, count in self.backlog_by_country.items()},
                **{f'backlog_{cat.value}': count for cat, count in self.backlog_by_eb_category.items()},
                **{f'backlog_{cat.value}_{nationality}': count 
                   for (cat, nationality), count in self.backlog_by_category_nationality.items()}
            }
        
        rows = []
        
        rows.append({
            'scenario': self.scenario_name,
            'category': 'TOTAL',
            'nationality': 'ALL',
            'backlog_size': self.total_backlog
        })
        
        for country, backlog in self.backlog_by_country.items():
            rows.append({
                'scenario': self.scenario_name,
                'category': 'ALL_CATEGORIES',
                'nationality': country,
                'backlog_size': backlog
            })
        
        for category, backlog in self.backlog_by_eb_category.items():
            rows.append({
                'scenario': self.scenario_name,
                'category': category.value,
                'nationality': 'ALL',
                'backlog_size': backlog
            })
        
        for (category, nationality), backlog in self.backlog_by_category_nationality.items():
            rows.append({
                'scenario': self.scenario_name,
                'category': category.value,
                'nationality': nationality,
                'backlog_size': backlog
            })
        
        return pd.DataFrame(rows)
    
    @classmethod
    def from_simulation(cls, sim: 'Simulation', scenario_name: str) -> 'BacklogAnalysis':
        """Create backlog analysis from completed simulation."""
        final_state = sim.states[-1]
        
        total_backlog = sum(final_state.queue_backlog_by_country.values())
        
        return cls(
            scenario_name=scenario_name,
            total_backlog=total_backlog,
            backlog_by_country=final_state.queue_backlog_by_country.copy(),
            backlog_by_eb_category=final_state.queue_backlog_by_eb_category.copy(),
            backlog_by_category_nationality=final_state.queue_backlog_by_eb_category_nationality.copy()
        )

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
