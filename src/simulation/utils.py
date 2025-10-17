# src/simulation/utils.py
"""
Utility functions for the workforce simulation.
Handles I/O, data serialization, and live data fetching.
Updated: Added print_simulation_results and compute_slots_sequence functions.
"""
import csv
import json
import pickle
import logging
import math
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import requests
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

def print_simulation_results(simulation, output_csv: str = None) -> None:
    """
    Nicely print simulation summary. Keeps CLI lean and avoids circular imports.
    `simulation` is a Simulation object with states and summary statistics.
    """
    print("\nSIMULATION SUMMARY")
    print("=" * 50)
    
    if hasattr(simulation, 'states') and simulation.states:
        initial_state = simulation.states[0]
        final_state = simulation.states[-1]
        years_simulated = len(simulation.states) - 1
        
        print(f"Years simulated: {years_simulated}")
        print(f"Industry: Information Technology")
        print(f"Simulation mode: {'Agent-based' if simulation.config.agent_mode else 'Count-based'}")
        print(f"Initial workers: {initial_state.total_workers:,}")
        print(f"Final workers: {final_state.total_workers:,}")
        print(f"Total conversions: {simulation.cumulative_conversions:,}")
        print(f"Final avg wage: ${final_state.avg_wage_total:,.0f}")
        
        if hasattr(simulation, 'country_cap_enabled') and simulation.country_cap_enabled:
            print(f"Per-country cap: ENABLED (with redistribution)")
        else:
            print(f"Per-country cap: DISABLED")
    
    if output_csv:
        print(f"Results saved to: {output_csv}")
    print("=" * 50)

def save_simulation_results(states, output_path: str, include_nationality_columns: bool = True) -> None:
    """
    Save simulation results to CSV file.
    
    Args:
        states: List of SimulationState objects
        output_path: Path to output CSV file
        include_nationality_columns: Whether to include nationality data in CSV
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            'year', 'total_workers', 'permanent_workers', 'temporary_workers',
            'new_permanent', 'new_temporary', 'converted_temps',
            'avg_wage_total', 'avg_wage_permanent', 'avg_wage_temporary', 'total_wage_bill'
        ]
        
        # Add nationality columns if requested
        if include_nationality_columns:
            fieldnames.extend(['top_temp_nationalities'])
        
        # Add per-country cap columns if any state has them
        has_country_cap_data = any(hasattr(state, 'country_cap_enabled') and state.country_cap_enabled for state in states)
        if has_country_cap_data:
            fieldnames.extend(['converted_by_country', 'queue_backlog_by_country'])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for state in states:
            row = {
                'year': state.year,
                'total_workers': state.total_workers,
                'permanent_workers': state.permanent_workers,
                'temporary_workers': state.temporary_workers,
                'new_permanent': state.new_permanent,
                'new_temporary': state.new_temporary,
                'converted_temps': state.converted_temps,
                'avg_wage_total': f"{state.avg_wage_total:.2f}",
                'avg_wage_permanent': f"{state.avg_wage_permanent:.2f}",
                'avg_wage_temporary': f"{state.avg_wage_temporary:.2f}",
                'total_wage_bill': f"{state.total_wage_bill:.2f}"
            }
            
            # Add nationality data if requested
            if include_nationality_columns:
                nationality_str = json.dumps(state.top_temp_nationalities) if state.top_temp_nationalities else "{}"
                row['top_temp_nationalities'] = nationality_str
            
            # Add per-country cap data if available
            if has_country_cap_data:
                if hasattr(state, 'country_cap_enabled') and state.country_cap_enabled:
                    conversions_str = json.dumps(state.converted_by_country) if state.converted_by_country else "{}"
                    backlogs_str = json.dumps(state.queue_backlog_by_country) if state.queue_backlog_by_country else "{}"
                    row['converted_by_country'] = conversions_str
                    row['queue_backlog_by_country'] = backlogs_str
                else:
                    row['converted_by_country'] = "{}"
                    row['queue_backlog_by_country'] = "{}"
            
            writer.writerow(row)
    
    logger.info(f"Saved simulation results to {output_path}")

def save_backlog_analysis(backlog_analysis, output_path: str) -> None:
    """
    Save backlog analysis results to CSV file.
    
    Args:
        backlog_analysis: BacklogAnalysis object
        output_path: Path to output CSV file
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame and save
    df = backlog_analysis.to_dataframe()
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved backlog analysis: {output_path}")

def load_backlog_analysis(input_path: str) -> pd.DataFrame:
    """
    Load backlog analysis results from CSV file.
    
    Args:
        input_path: Path to input CSV file
    
    Returns:
        DataFrame with backlog analysis data
    """
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded backlog analysis from {input_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Backlog analysis file not found: {input_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading backlog analysis: {e}")
        return pd.DataFrame()

def validate_configuration(config) -> List[str]:
    """
    Validate simulation configuration parameters.
    
    Args:
        config: SimulationConfig to validate
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if config.initial_workers <= 0:
        errors.append("Initial workers must be positive")
    
    if config.years <= 0:
        errors.append("Simulation years must be positive")
    
    if config.years > 50:
        errors.append("Simulation years should not exceed 50 for performance reasons")
    
    if config.initial_workers > 1_000_000_000:
        errors.append("Initial workers seems unrealistically large")
    
    # Agent-mode performance warnings
    if config.agent_mode and config.initial_workers > 500000:
        errors.append("Agent-mode with >500K workers may be very slow. Consider count-mode.")
    
    # Per-country cap specific warnings
    if config.country_cap_enabled and config.initial_workers < 1000:
        errors.append("Per-country cap with <1000 workers may show high discretization effects")
    
    # Backlog comparison specific warnings
    if config.compare_backlogs and config.initial_workers < 5000:
        errors.append("Backlog comparison with <5000 workers may not show meaningful differences")
    
    if config.compare_backlogs and config.years < 10:
        errors.append("Backlog comparison with <10 years may not allow sufficient backlog accumulation")
    
    # Check output path is writable
    try:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = output_path.parent / ".test_write"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        errors.append(f"Cannot write to output path {config.output_path}: {e}")
    
    return errors

def format_number(num: int) -> str:
    """Format large numbers with commas for readability."""
    return f"{num:,}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format decimal as percentage with specified decimal places."""
    return f"{value:.{decimal_places}%}"

def format_currency(value: float, decimal_places: int = 0) -> str:
    """Format currency values."""
    return f"${value:,.{decimal_places}f}"

def print_data_sources(live_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Print data sources and citations to stdout.
    
    Args:
        live_data: Optional live data dictionary with sources
    """
    print("\n" + "="*60)
    print("DATA SOURCES AND CITATIONS")
    print("="*60)
    
    if live_data and "sources" in live_data:
        print("\nLive data sources (fetched at runtime):")
        for source in live_data["sources"]:
            print(f"  • {source}")
        print(f"\nData timestamp: {live_data.get('data_timestamp', 'Unknown')}")
    else:
        print("\nDefault empirical data sources:")
        print("  • U.S. Bureau of Labor Statistics Employment Situation August 2025")
        print("  • BLS Job Openings and Labor Turnover Survey (JOLTS) 2024")
        print("  • BLS Occupational Employment Statistics IT Sector 2024")
        print("  • USCIS H-1B Visa FY 2024 Reports and Data")
        print("  • USCIS H-1B Nationality Distribution FY 2024")
        print("  • DOL H-1B Disclosure Data by Country of Birth 2024")
        print("  • USCIS Employment-Based Green Card FY 2024 Reports")
        print("  • Immigration and Nationality Act Section 203(b) Per-Country Limitation")
        print("  • USCIS Green Card Backlog Reports by Country 2024")
        print("  • Jennifer Hunt Research on Temporary Worker Job Mobility")
        print("  • American Immigration Council H-1B Analysis 2024")
        print("  • National Foundation for American Policy H-1B Analysis 2024")
    
    print("="*60)
