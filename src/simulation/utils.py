# src/simulation/utils.py
"""
Utility functions for immigration simulation.
UPDATED: Added family-adjusted backlog export to CSV.
FIXED: save_backlog_analysis() now handles both single and list inputs.
"""
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Union
from .models import SimulationState, BacklogAnalysis

logger = logging.getLogger(__name__)


def save_states_to_csv(states: List[SimulationState], output_path: str, dependent_children: List = None) -> None:
    """
    Save simulation states to CSV file with family-adjusted backlog calculations.

    Args:
        states: List of simulation states
        output_path: Path to output CSV file
        dependent_children: List of dependent children (for family backlog calculation)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'year', 'total_workers', 'permanent_workers', 'temporary_workers',
            'new_permanent', 'new_temporary', 'converted_temps',
            'avg_wage_total', 'avg_wage_permanent', 'avg_wage_temporary',
            'total_wage_bill', 'cumulative_conversions', 'h1b_share',
            'children_aged_out_this_year', 'cumulative_children_aged_out', 'children_at_risk',
            'queue_backlog_total', 'family_adjusted_backlog_total',
            'country_cap_enabled', 'annual_conversion_cap'
        ]

        # Add per-nationality backlog columns
        nationalities = set()
        for state in states:
            nationalities.update(state.queue_backlog_by_country.keys())

        for nat in sorted(nationalities):
            fieldnames.append(f'backlog_{nat}')
            fieldnames.append(f'family_backlog_{nat}')

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for state in states:
            # Calculate family-adjusted backlog for this state
            if dependent_children is not None:
                # Filter children to those existing at this state's year
                current_children = [c for c in dependent_children if c.entry_year <= state.year]
                family_backlog = state.calculate_family_adjusted_backlog_with_children(current_children)
                total_family_backlog = sum(family_backlog.values())
            else:
                family_backlog = state.calculate_family_adjusted_backlog()
                total_family_backlog = sum(family_backlog.values())

            row = {
                'year': state.year,
                'total_workers': state.total_workers,
                'permanent_workers': state.permanent_workers,
                'temporary_workers': state.temporary_workers,
                'new_permanent': state.new_permanent,
                'new_temporary': state.new_temporary,
                'converted_temps': state.converted_temps,
                'avg_wage_total': round(state.avg_wage_total, 2),
                'avg_wage_permanent': round(state.avg_wage_permanent, 2),
                'avg_wage_temporary': round(state.avg_wage_temporary, 2),
                'total_wage_bill': round(state.total_wage_bill, 2),
                'cumulative_conversions': state.cumulative_conversions,
                'h1b_share': round(state.h1b_share, 4),
                'children_aged_out_this_year': state.children_aged_out_this_year,
                'cumulative_children_aged_out': state.cumulative_children_aged_out,
                'children_at_risk': state.children_at_risk,
                'queue_backlog_total': sum(state.queue_backlog_by_country.values()),
                'family_adjusted_backlog_total': total_family_backlog,
                'country_cap_enabled': state.country_cap_enabled,
                'annual_conversion_cap': state.annual_conversion_cap
            }

            # Add per-nationality backlogs
            for nat in sorted(nationalities):
                principal_backlog = state.queue_backlog_by_country.get(nat, 0)
                family_backlog_nat = family_backlog.get(nat, 0)
                row[f'backlog_{nat}'] = principal_backlog
                row[f'family_backlog_{nat}'] = family_backlog_nat

            writer.writerow(row)

    logger.info(f"Saved simulation results to {output_file}")


def save_backlog_analysis(analyses: Union[BacklogAnalysis, List[BacklogAnalysis]], output_path: str) -> None:
    """
    Save backlog analysis comparison to CSV file.

    Args:
        analyses: Single BacklogAnalysis or list of backlog analyses (e.g., capped vs uncapped)
        output_path: Path to output CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Handle both single BacklogAnalysis and list of BacklogAnalysis
    if isinstance(analyses, BacklogAnalysis):
        analyses = [analyses]

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'scenario', 'category', 'nationality', 
            'backlog_size', 'family_adjusted_backlog'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for analysis in analyses:
            df = analysis.to_dataframe()
            for _, row in df.iterrows():
                writer.writerow(row.to_dict())

    logger.info(f"Saved backlog analysis to {output_file}")


def load_results(input_path: str) -> List[Dict[str, Any]]:
    """
    Load simulation results from CSV file.

    Args:
        input_path: Path to input CSV file

    Returns:
        List of dictionaries containing simulation data
    """
    input_file = Path(input_path)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return []

    results = []
    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert numeric fields
            for key, value in row.items():
                if key == 'year' or key.startswith('backlog_') or key.startswith('family_backlog_'):
                    try:
                        row[key] = int(value) if value else 0
                    except ValueError:
                        pass
                elif key in ['h1b_share', 'avg_wage_total', 'avg_wage_permanent', 'avg_wage_temporary']:
                    try:
                        row[key] = float(value) if value else 0.0
                    except ValueError:
                        pass

            results.append(row)

    logger.info(f"Loaded {len(results)} rows from {input_file}")
    return results


def print_backlog_comparison(analyses: Union[BacklogAnalysis, List[BacklogAnalysis]]) -> None:
    """
    Print a formatted comparison of backlog analyses.

    Args:
        analyses: Single BacklogAnalysis or list of backlog analyses to compare
    """
    # Handle both single BacklogAnalysis and list of BacklogAnalysis
    if isinstance(analyses, BacklogAnalysis):
        analyses = [analyses]

    if not analyses:
        logger.warning("No backlog analyses to compare")
        return

    print("\n" + "="*80)
    print("BACKLOG ANALYSIS COMPARISON")
    print("="*80)

    for analysis in analyses:
        print(f"\n{analysis.scenario_name}:")
        print(f"  Total Backlog (principals): {analysis.total_backlog:,}")
        print(f"  Total Family-Adjusted Backlog: {analysis.total_family_adjusted_backlog:,}")

        if analysis.total_backlog > 0:
            multiplier = analysis.total_family_adjusted_backlog / analysis.total_backlog
            print(f"  Family Multiplier: {multiplier:.2f}x")
        else:
            print(f"  Family Multiplier: N/A (no backlog)")

        print("\n  By Nationality (Principal | Family-Adjusted):")

        for nationality in sorted(analysis.backlog_by_country.keys()):
            principal = analysis.backlog_by_country[nationality]
            family = analysis.family_adjusted_backlog.get(nationality, 0)
            if principal > 0 or family > 0:
                print(f"    {nationality:10s}: {principal:6,} | {family:6,}")

    print("\n" + "="*80)
