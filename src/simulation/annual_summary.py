# src/simulation/annual_summary.py
"""
Clean annual summary generator for immigration simulation results.
Provides structured, readable summaries without verbose debug output.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from .models import SimulationState, EBCategory

logger = logging.getLogger(__name__)


@dataclass
class AnnualSummary:
    """Clean annual summary of key simulation metrics."""
    year: int
    workforce_size: int
    permanent_workers: int
    temporary_workers: int
    h1b_share: float
    new_h1b_added: int
    new_permanent_added: int
    total_workers_added: int
    conversions: int
    children_aged_out_this_year: int
    cumulative_children_aged_out: int
    avg_wage: float
    total_queue_backlog: int
    
    # EB category breakdowns
    conversions_eb1: int
    conversions_eb2: int
    conversions_eb3: int
    
    # Nationality breakdowns
    india_backlog: int
    china_backlog: int
    other_backlog: int


def generate_annual_summaries(states: List[SimulationState], scenario_name: str = "") -> List[AnnualSummary]:
    """
    Generate clean annual summaries from simulation states.
    
    Args:
        states: List of simulation states (one per year)
        scenario_name: Name of scenario for logging
        
    Returns:
        List of AnnualSummary objects, one per year
    """
    summaries = []
    
    for i, state in enumerate(states):
        # Calculate queue backlog
        total_queue_backlog = sum(state.queue_backlog_by_country.values())
        
        # Get EB category conversions
        conversions_eb1 = state.converted_by_eb_category.get(EBCategory.EB1, 0)
        conversions_eb2 = state.converted_by_eb_category.get(EBCategory.EB2, 0)
        conversions_eb3 = state.converted_by_eb_category.get(EBCategory.EB3, 0)
        
        # Get nationality backlogs
        india_backlog = state.queue_backlog_by_country.get('India', 0)
        china_backlog = state.queue_backlog_by_country.get('China', 0)
        other_backlog = state.queue_backlog_by_country.get('Other', 0)
        
        summary = AnnualSummary(
            year=state.year,
            workforce_size=state.total_workers,
            permanent_workers=state.permanent_workers,
            temporary_workers=state.temporary_workers,
            h1b_share=state.h1b_share,
            new_h1b_added=state.new_temporary,
            new_permanent_added=state.new_permanent,
            total_workers_added=state.new_permanent + state.new_temporary,
            conversions=state.converted_temps,
            children_aged_out_this_year=state.children_aged_out_this_year,
            cumulative_children_aged_out=state.cumulative_children_aged_out,
            avg_wage=state.avg_wage_total,
            total_queue_backlog=total_queue_backlog,
            conversions_eb1=conversions_eb1,
            conversions_eb2=conversions_eb2,
            conversions_eb3=conversions_eb3,
            india_backlog=india_backlog,
            china_backlog=china_backlog,
            other_backlog=other_backlog
        )
        
        summaries.append(summary)
    
    return summaries


def print_annual_summary_table(summaries: List[AnnualSummary], scenario_name: str = ""):
    """
    Print a clean annual summary table to console.
    
    Args:
        summaries: List of annual summaries
        scenario_name: Name of scenario for table header
    """
    if not summaries:
        return
    
    # Print header
    title = f"ANNUAL SUMMARY: {scenario_name.upper()}" if scenario_name else "ANNUAL SUMMARY"
    print()
    print("=" * 140)
    print(title)
    print("=" * 140)
    
    # Print column headers
    header = (f"{'Year':>4} | {'Workforce':>9} | {'Perm':>8} | {'H-1B':>8} | {'H-1B%':>6} | "
             f"{'New H-1B':>8} | {'New Perm':>8} | {'Conv':>6} | {'EB1':>4} | {'EB2':>4} | {'EB3':>4} | "
             f"{'Kids Out':>8} | {'Cum Kids':>8} | {'Avg Wage':>9} | {'Queue':>8}")
    print(header)
    print("-" * 140)
    
    # Print data rows
    for summary in summaries:
        row = (f"{summary.year:4d} | {summary.workforce_size:9,d} | {summary.permanent_workers:8,d} | "
               f"{summary.temporary_workers:8,d} | {summary.h1b_share:6.2%} | {summary.new_h1b_added:8,d} | "
               f"{summary.new_permanent_added:8,d} | {summary.conversions:6,d} | {summary.conversions_eb1:4d} | "
               f"{summary.conversions_eb2:4d} | {summary.conversions_eb3:4d} | {summary.children_aged_out_this_year:8,d} | "
               f"{summary.cumulative_children_aged_out:8,d} | ${summary.avg_wage:9,.0f} | {summary.total_queue_backlog:8,d}")
        print(row)
    
    print("-" * 140)
    print("Columns: Year | Total Workforce | Permanent | H-1B | H-1B Share | New H-1B | New Permanent | Conversions | EB-1 | EB-2 | EB-3 | Children Aged Out | Cumulative Children | Average Wage | Queue Backlog")
    print("=" * 140)


def print_nationality_backlog_summary(summaries: List[AnnualSummary], scenario_name: str = "", years_to_show: List[int] = None):
    """
    Print nationality-specific backlog summary for selected years.
    
    Args:
        summaries: List of annual summaries
        scenario_name: Name of scenario for table header
        years_to_show: List of specific years to show (default: every 10 years + final year)
    """
    if not summaries:
        return
    
    # Determine years to show
    if years_to_show is None:
        first_year = summaries[0].year
        last_year = summaries[-1].year
        years_to_show = list(range(first_year, last_year + 1, 10))
        if last_year not in years_to_show:
            years_to_show.append(last_year)
    
    # Filter summaries to requested years
    filtered_summaries = [s for s in summaries if s.year in years_to_show]
    
    if not filtered_summaries:
        return
    
    # Print header
    title = f"NATIONALITY BACKLOG SUMMARY: {scenario_name.upper()}" if scenario_name else "NATIONALITY BACKLOG SUMMARY"
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    
    # Print column headers
    header = f"{'Year':>4} | {'India':>10} | {'China':>10} | {'Other':>10} | {'Total':>10}"
    print(header)
    print("-" * 80)
    
    # Print data rows
    for summary in filtered_summaries:
        row = (f"{summary.year:4d} | {summary.india_backlog:10,d} | {summary.china_backlog:10,d} | "
               f"{summary.other_backlog:10,d} | {summary.total_queue_backlog:10,d}")
        print(row)
    
    print("=" * 80)


def save_annual_summaries_csv(summaries: List[AnnualSummary], output_path: str, scenario_name: str = ""):
    """
    Save annual summaries to CSV file.
    
    Args:
        summaries: List of annual summaries
        output_path: Path to save CSV file
        scenario_name: Scenario name to include in CSV
    """
    import csv
    from pathlib import Path
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'scenario', 'year', 'workforce_size', 'permanent_workers', 'temporary_workers', 
            'h1b_share', 'new_h1b_added', 'new_permanent_added', 'total_workers_added',
            'conversions', 'conversions_eb1', 'conversions_eb2', 'conversions_eb3',
            'children_aged_out_this_year', 'cumulative_children_aged_out',
            'avg_wage', 'total_queue_backlog', 'india_backlog', 'china_backlog', 'other_backlog'
        ])
        
        # Write data
        for summary in summaries:
            writer.writerow([
                scenario_name, summary.year, summary.workforce_size, summary.permanent_workers,
                summary.temporary_workers, summary.h1b_share, summary.new_h1b_added,
                summary.new_permanent_added, summary.total_workers_added, summary.conversions,
                summary.conversions_eb1, summary.conversions_eb2, summary.conversions_eb3,
                summary.children_aged_out_this_year, summary.cumulative_children_aged_out,
                summary.avg_wage, summary.total_queue_backlog, summary.india_backlog,
                summary.china_backlog, summary.other_backlog
            ])
    
    logger.info(f"Saved annual summaries to {output_path}")


def compare_scenarios_summary(summaries_uncapped: List[AnnualSummary], 
                            summaries_capped: List[AnnualSummary],
                            years_to_compare: List[int] = None):
    """
    Print a side-by-side comparison of key metrics between scenarios.
    
    Args:
        summaries_uncapped: Annual summaries for uncapped scenario
        summaries_capped: Annual summaries for capped scenario  
        years_to_compare: Specific years to compare (default: every 10 years + final year)
    """
    if not summaries_uncapped or not summaries_capped:
        return
    
    # Determine years to compare
    if years_to_compare is None:
        first_year = max(summaries_uncapped[0].year, summaries_capped[0].year)
        last_year = min(summaries_uncapped[-1].year, summaries_capped[-1].year)
        years_to_compare = list(range(first_year, last_year + 1, 10))
        if last_year not in years_to_compare:
            years_to_compare.append(last_year)
    
    # Create lookup dictionaries
    uncapped_lookup = {s.year: s for s in summaries_uncapped}
    capped_lookup = {s.year: s for s in summaries_capped}
    
    print()
    print("=" * 120)
    print("SCENARIO COMPARISON: UNCAPPED vs CAPPED")
    print("=" * 120)
    
    # Print header
    header = (f"{'Year':>4} | {'Uncapped':>15} | {'Capped':>15} | {'Difference':>12} | "
             f"{'Metric':>20}")
    print(header)
    print("-" * 120)
    
    for year in years_to_compare:
        if year not in uncapped_lookup or year not in capped_lookup:
            continue
        
        uncapped = uncapped_lookup[year]
        capped = capped_lookup[year]
        
        # Compare key metrics
        metrics = [
            ("Conversions", uncapped.conversions, capped.conversions),
            ("Children Aged Out", uncapped.children_aged_out_this_year, capped.children_aged_out_this_year),
            ("Cumulative Aged Out", uncapped.cumulative_children_aged_out, capped.cumulative_children_aged_out),
            ("Queue Backlog", uncapped.total_queue_backlog, capped.total_queue_backlog),
            ("H-1B Share", f"{uncapped.h1b_share:.2%}", f"{capped.h1b_share:.2%}", "N/A")
        ]
        
        for i, (metric_name, uncapped_val, capped_val, *diff_override) in enumerate(metrics):
            if i == 0:  # First row shows year
                year_str = f"{year:4d}"
            else:
                year_str = "    "
            
            if diff_override:
                difference_str = diff_override[0]
            elif isinstance(uncapped_val, str):
                difference_str = "N/A"
            else:
                difference = uncapped_val - capped_val
                difference_str = f"{difference:+,d}"
            
            row = (f"{year_str} | {str(uncapped_val):>15} | {str(capped_val):>15} | "
                   f"{difference_str:>12} | {metric_name:>20}")
            print(row)
        
        if year != years_to_compare[-1]:  # Add separator except for last year
            print(" " * 120)
    
    print("=" * 120)
