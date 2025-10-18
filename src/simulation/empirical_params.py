# src/simulation/empirical_params.py
"""
Empirical parameters for immigration simulation.
UPDATED: Added spouse multiplier constant for family-adjusted backlog calculations.
"""
from enum import Enum
from typing import Dict, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Core workforce parameters
H1B_SHARE = 0.0045  # 0.45% of workforce are H-1B holders
ANNUAL_WORKFORCE_GROWTH_RATE = 0.005  # 0.5% annual growth
REAL_US_WORKFORCE_SIZE = 165_000_000  # Real US workforce size
GREEN_CARD_CAP_ABS = 63_000  # Annual employment-based green card cap (principals only)

# Per-country cap parameters
PER_COUNTRY_CAP_SHARE = 0.07  # 7% per-country limit

# Worker lifecycle parameters
STARTING_WAGE = 95_000  # Starting salary for new workers
CONVERSION_WAGE_BUMP = 1.05  # 5% wage increase on conversion
PERM_FILING_DELAY = 2  # Years between H-1B entry and PERM filing

# Job mobility parameters
JOB_CHANGE_PROB_PERM = 0.12  # 12% annual job change probability for permanent workers
JOB_CHANGE_PROB_TEMP = 0.096  # 9.6% for temporary (20% less mobile)

# Wage jump parameters on job change
WAGE_JUMP_FACTOR_MEAN_PERM = 1.50  # 15% average wage jump for permanent workers
WAGE_JUMP_FACTOR_STD_PERM = 0.08  # 8% standard deviation
WAGE_JUMP_FACTOR_MEAN_TEMP = 1.12  # 8% average wage jump for temporary workers
WAGE_JUMP_FACTOR_STD_TEMP = 0.06  # 6% standard deviation

# Nationality distribution for H-1B workers
TEMP_NATIONALITY_DISTRIBUTION = {
    "India": 0.73,  # 73% of H-1B workers
    "China": 0.12,  # 12% of H-1B workers
    "Other": 0.15   # 15% from other countries
}

# Permanent workers predominantly US nationality
PERMANENT_NATIONALITY = "USA"

# Carryover strategy
CARRYOVER_FRACTION_STRATEGY = True

from .models import EBCategory

# EB category statutory shares (from US law)
EB_CATEGORY_STATUTORY_SHARES = {
    EBCategory.EB1: 0.286,  # 28.6% for priority workers
    EBCategory.EB2: 0.286,  # 28.6% for advanced degree professionals
    EBCategory.EB3: 0.286,  # 28.6% for skilled workers
    EBCategory.EB4: 0.071,  # 7.1% for special immigrants
    EBCategory.EB5: 0.071   # 7.1% for investors
}

# H-1B to EB category conversion probabilities (empirically derived)
H1B_CONVERSION_CATEGORY_PROBABILITIES = {
    EBCategory.EB1: 0.05,  # 5% convert through EB-1 (priority workers)
    EBCategory.EB2: 0.70,  # 70% convert through EB-2 (advanced degree)
    EBCategory.EB3: 0.25,  # 25% convert through EB-3 (skilled workers)
    EBCategory.EB4: 0.00,  # 0% convert through EB-4 (special immigrants)
    EBCategory.EB5: 0.00   # 0% convert through EB-5 (investors)
}

# Child age-out parameters for continuous aging
CHILDREN_PER_H1B_WORKER = 2.0  # Average children per H-1B worker
CHILD_ENTRY_AGE_MEAN = 12.0  # Average age when child enters with H-1B parent
CHILD_ENTRY_AGE_STD = 4.0  # Standard deviation of entry age
CHILD_AGEOUT_AGE = 21  # Age at which children age out

# Family backlog calculation parameters (NEW)
SPOUSE_MULTIPLIER = 2.0  # Multiplier for principals to account for spouses (principal + spouse = 2)
# Family-adjusted backlog = (principals × SPOUSE_MULTIPLIER) + dependent_children

def calculate_annual_eb_caps(initial_workers: int) -> Dict[EBCategory, int]:
    """
    Calculate annual EB category caps based on simulation size.
    FIXED: Ensure total never exceeds realistic limits.
    """
    scaling_factor = initial_workers / REAL_US_WORKFORCE_SIZE
    total_sim_cap = round(GREEN_CARD_CAP_ABS * scaling_factor)

    eb_caps = {}
    for category, share in EB_CATEGORY_STATUTORY_SHARES.items():
        eb_caps[category] = max(1, round(total_sim_cap * share))

    # Allocate rounding difference to largest category
    actual_total = sum(eb_caps.values())
    if actual_total != total_sim_cap:
        difference = total_sim_cap - actual_total
        largest_category = max(eb_caps.keys(), key=lambda cat: eb_caps[cat])
        eb_caps[largest_category] = max(1, eb_caps[largest_category] + difference)

    logger.info(f"EB caps calculated: total={sum(eb_caps.values())}, breakdown={eb_caps}")
    return eb_caps


def calculate_per_country_caps_by_category(annual_eb_caps: Dict[EBCategory, int]) -> Dict[EBCategory, Dict[str, int]]:
    """Calculate per-country caps within each EB category (7% rule)."""
    per_country_caps = {}

    for category, total_cap in annual_eb_caps.items():
        category_per_country_cap = max(1, round(total_cap * PER_COUNTRY_CAP_SHARE))
        per_country_caps[category] = {
            "India": category_per_country_cap,
            "China": category_per_country_cap,
            "Other": total_cap  # Other countries not subject to per-country limits (only limited by total green cards issued in a year)
        }

    return per_country_caps


def calculate_fixed_h1b_entries(initial_workers: int) -> int:
    """
    FIXED: Calculate realistic H-1B entries that create sustainable backlogs.
    """
    scaling_factor = initial_workers / REAL_US_WORKFORCE_SIZE

    # CRITICAL: Based on actual H-1B approval rates (~85,000 annually)
    real_annual_h1b_entries = 85_000
    sim_annual_h1b_entries = round(real_annual_h1b_entries * scaling_factor)

    logger.info(f"H-1B entries calculated: {sim_annual_h1b_entries}/year")
    return sim_annual_h1b_entries


def calculate_annual_sim_cap(initial_workers: int) -> int:
    """Calculate total annual simulation cap (legacy compatibility)."""
    return sum(calculate_annual_eb_caps(initial_workers).values())


def calculate_permanent_entries(current_total: int, new_temporary: int, scenario: str = "base") -> int:
    """
    Calculate new permanent worker entries to maintain workforce growth.

    Args:
        current_total: Current total workforce size
        new_temporary: Number of new temporary workers being added
        scenario: Scenario identifier for consistency

    Returns:
        Number of new permanent workers to add
    """
    # Target workforce size based on growth rate
    target_total = round(current_total * (1 + ANNUAL_WORKFORCE_GROWTH_RATE))

    # New permanent workers = target growth - new temporary workers
    new_permanent = max(0, target_total - current_total - new_temporary)

    return new_permanent
