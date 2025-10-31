# src/simulation/cli.py
"""
Command-line interface for the workforce simulation.
CLEANUP: Streamlined to work with cleaned up models and removed dependencies.
MAJOR UPDATE: Added EB-1 through EB-5 category support with enhanced visualizations.
REFACTORED: Moved all plotting logic to visualization.py
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import time  # For measuring total runtime


from .models import SimulationConfig, BacklogAnalysis, EBCategory
from .sim import Simulation
from .utils import save_backlog_analysis
from .visualization import SimulationVisualizer


logger = logging.getLogger(__name__)



def setup_logging(debug: bool = False):
    """Setup logging configuration with font debug suppression."""
    level = logging.DEBUG if debug else logging.INFO


    # Configure main logging
    logging.basicConfig(
        level=level,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


    # Suppress matplotlib font manager verbose logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


    # Suppress other verbose libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)



def create_config_from_args(args) -> SimulationConfig:
    """Create SimulationConfig from command line arguments."""
    return SimulationConfig(
        initial_workers=args.initial_workers,
        years=args.years,
        seed=args.seed,
        output_path=args.output,
        country_cap_enabled=args.country_cap,
        compare_backlogs=args.compare,
        debug=args.debug,
        start_year=args.start_year
    )



def save_simulation_results_csv(states, filepath: str):
    """Save simulation results to CSV file with EB category data."""
    import csv
    from pathlib import Path


    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)


        # Write header - UPDATED to include EB category data
        writer.writerow([
            'year', 'total_workers', 'permanent_workers', 'temporary_workers',
            'new_permanent', 'new_temporary', 'converted_temps',
            'avg_wage_total', 'avg_wage_permanent', 'avg_wage_temporary',
            'total_wage_bill', 'h1b_share', 'cumulative_conversions',
            'children_aged_out_this_year', 'cumulative_children_aged_out',
            'converted_eb1', 'converted_eb2', 'converted_eb3', 'converted_eb4', 'converted_eb5',
            'backlog_eb1', 'backlog_eb2', 'backlog_eb3', 'backlog_eb4', 'backlog_eb5',
            'backlog_eb2_india', 'backlog_eb2_china', 'backlog_eb2_other',
            'backlog_eb3_india', 'backlog_eb3_china', 'backlog_eb3_other',
            'country_cap_enabled', 'total_wages'
        ])


        # Write data rows
        for state in states:
            writer.writerow([
                state.year, state.total_workers, state.permanent_workers, state.temporary_workers,
                state.new_permanent, state.new_temporary, state.converted_temps,
                state.avg_wage_total, state.avg_wage_permanent, state.avg_wage_temporary,
                state.total_wage_bill, state.h1b_share, state.cumulative_conversions,
                state.children_aged_out_this_year, state.cumulative_children_aged_out,
                # EB category conversions
                state.converted_by_eb_category.get(EBCategory.EB1, 0),
                state.converted_by_eb_category.get(EBCategory.EB2, 0),
                state.converted_by_eb_category.get(EBCategory.EB3, 0),
                state.converted_by_eb_category.get(EBCategory.EB4, 0),
                state.converted_by_eb_category.get(EBCategory.EB5, 0),
                # EB category backlogs
                state.queue_backlog_by_eb_category.get(EBCategory.EB1, 0),
                state.queue_backlog_by_eb_category.get(EBCategory.EB2, 0),
                state.queue_backlog_by_eb_category.get(EBCategory.EB3, 0),
                state.queue_backlog_by_eb_category.get(EBCategory.EB4, 0),
                state.queue_backlog_by_eb_category.get(EBCategory.EB5, 0),
                # Detailed country-category backlogs
                state.queue_backlog_by_eb_category_nationality.get((EBCategory.EB2, 'India'), 0),
                state.queue_backlog_by_eb_category_nationality.get((EBCategory.EB2, 'China'), 0),
                state.queue_backlog_by_eb_category_nationality.get((EBCategory.EB2, 'Other'), 0),
                state.queue_backlog_by_eb_category_nationality.get((EBCategory.EB3, 'India'), 0),
                state.queue_backlog_by_eb_category_nationality.get((EBCategory.EB3, 'China'), 0),
                state.queue_backlog_by_eb_category_nationality.get((EBCategory.EB3, 'Other'), 0),
                state.country_cap_enabled,
                state.total_wages
            ])



def run_single_simulation(config: SimulationConfig) -> None:
    """Run a single simulation with the given configuration."""
    logger.info(f"Running simulation: {config.initial_workers:,} workers, {config.years} years")


    # Runtime measurement
    t0 = time.perf_counter()


    # Create and run simulation
    sim = Simulation(config)
    states = sim.run()


    elapsed = time.perf_counter() - t0
    print(f"\nRuntime: {elapsed:0.3f} seconds")


    # Save results
    try:
        save_simulation_results_csv(states, config.output_path)
        logger.info(f"Results saved to: {config.output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise



def run_comparative_analysis(config: SimulationConfig) -> None:
    """Run comparative analysis between capped and uncapped scenarios with EB category awareness."""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS (No Cap vs 7% Per-Country Cap Within Each EB Category)")
    print("="*80)


    total_start = time.perf_counter()  # total runtime


    # Run uncapped simulation
    print("\n[1/3] Running uncapped simulation...")
    config_uncapped = SimulationConfig(
        initial_workers=config.initial_workers,
        years=config.years,
        seed=config.seed,
        country_cap_enabled=False,
        debug=config.debug,
        start_year=config.start_year
    )
    t0_uncapped = time.perf_counter()
    sim_uncapped = Simulation(config_uncapped)
    states_uncapped = sim_uncapped.run()
    uncapped_runtime = time.perf_counter() - t0_uncapped
    print(f"Uncapped runtime: {uncapped_runtime:0.3f} seconds")


    # Run capped simulation
    print("\n[2/3] Running capped simulation (7% per-country cap within each EB category)...")
    config_capped = SimulationConfig(
        initial_workers=config.initial_workers,
        years=config.years,
        seed=config.seed,
        country_cap_enabled=True,
        debug=config.debug,
        start_year=config.start_year
    )
    t0_capped = time.perf_counter()
    sim_capped = Simulation(config_capped)
    states_capped = sim_capped.run()
    capped_runtime = time.perf_counter() - t0_capped
    print(f"Capped runtime: {capped_runtime:0.3f} seconds")


    # Generate comparison report AFTER both simulations are complete
    print("\n[3/3] Generating conversion comparison analysis...")
    df_comparison = generate_conversion_comparison_report(states_uncapped, states_capped, Path(config.output_path).parent)


    # Save individual results
    uncapped_path = config.output_path.replace('.csv', '_uncapped.csv')
    capped_path = config.output_path.replace('.csv', '_capped.csv')


    try:
        save_simulation_results_csv(states_uncapped, uncapped_path)
        save_simulation_results_csv(states_capped, capped_path)
        logger.info(f"Uncapped results saved to: {uncapped_path}")
        logger.info(f"Capped results saved to: {capped_path}")
    except Exception as e:
        logger.error(f"Error saving simulation results: {e}")
        raise


    # Generate comparative analysis
    try:
        backlog_uncapped = BacklogAnalysis.from_simulation(sim_uncapped, "uncapped")
        backlog_capped = BacklogAnalysis.from_simulation(sim_capped, "capped")


        # Save backlog analyses using the utility function
        backlog_uncapped_path = config.output_path.replace('.csv', '_backlog_uncapped.csv')
        backlog_capped_path = config.output_path.replace('.csv', '_backlog_capped.csv')


        save_backlog_analysis(backlog_uncapped, backlog_uncapped_path)
        save_backlog_analysis(backlog_capped, backlog_capped_path)


        logger.info(f"Backlog analysis saved to: {backlog_uncapped_path}")
        logger.info(f"Backlog analysis saved to: {backlog_capped_path}")


        # Generate visualizations using visualization module
        logger.info("Generating EB category comparison charts...")
        try:
            output_dir = Path(config.output_path).parent
            visualizer = SimulationVisualizer(output_dir=str(output_dir), save_plots=True)
            chart_path = visualizer.generate_eb_category_comparison_charts(
                states_uncapped, states_capped, backlog_uncapped, backlog_capped
            )
            logger.info(f"✅ EB category comparison charts generated successfully: {chart_path}")
            
        except ImportError as e:
            logger.error(f"❌ Chart generation failed - missing dependencies: {e}")
            logger.error("Install matplotlib and seaborn: pip install matplotlib seaborn")
        except Exception as e:
            logger.error(f"❌ Chart generation failed: {e}")
            if config.debug:
                import traceback
                traceback.print_exc()


        # Print comparison summary
        print_eb_category_comparison_summary(states_uncapped, states_capped, backlog_uncapped, backlog_capped)


    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        raise


    total_elapsed = time.perf_counter() - total_start
    print("\n" + "-"*80)
    print(f"Total comparative analysis runtime: {total_elapsed:0.3f} seconds")
    print("-"*80)



def generate_conversion_comparison_report(states_uncapped, states_capped, output_dir: Path):
    """
    Generate a detailed comparison report of annual conversions between scenarios.
    This will help identify if spillover is equalizing the scenarios.
    """
    import pandas as pd


    # Extract conversion data from both scenarios
    comparison_data = []


    for i, (uncapped_state, capped_state) in enumerate(zip(states_uncapped, states_capped)):
        if i == 0:  # Skip initial state
            continue


        year = uncapped_state.year


        # Total conversions
        uncapped_total = uncapped_state.converted_temps
        capped_total = capped_state.converted_temps


        # Conversions by country
        uncapped_india = uncapped_state.converted_by_country.get('India', 0)
        uncapped_china = uncapped_state.converted_by_country.get('China', 0)
        uncapped_other = uncapped_state.converted_by_country.get('Other', 0)


        capped_india = capped_state.converted_by_country.get('India', 0)
        capped_china = capped_state.converted_by_country.get('China', 0)
        capped_other = capped_state.converted_by_country.get('Other', 0)


        # Children aged out
        uncapped_aged_out = uncapped_state.children_aged_out_this_year
        capped_aged_out = capped_state.children_aged_out_this_year


        # Backlogs
        uncapped_india_backlog = uncapped_state.queue_backlog_by_country.get('India', 0)
        uncapped_china_backlog = uncapped_state.queue_backlog_by_country.get('China', 0)


        capped_india_backlog = capped_state.queue_backlog_by_country.get('India', 0)
        capped_china_backlog = capped_state.queue_backlog_by_country.get('China', 0)


        # Calculate differences
        total_conversion_diff = uncapped_total - capped_total
        total_conversion_ratio = uncapped_total / capped_total if capped_total > 0 else float('inf')


        comparison_data.append({
            'Year': year,
            'Uncapped_Total_Conversions': uncapped_total,
            'Capped_Total_Conversions': capped_total,
            'Conversion_Difference': total_conversion_diff,
            'Conversion_Ratio': total_conversion_ratio,
            'Uncapped_India_Conversions': uncapped_india,
            'Capped_India_Conversions': capped_india,
            'Uncapped_China_Conversions': uncapped_china,
            'Capped_China_Conversions': capped_china,
            'Uncapped_Other_Conversions': uncapped_other,
            'Capped_Other_Conversions': capped_other,
            'Uncapped_Children_Aged_Out': uncapped_aged_out,
            'Capped_Children_Aged_Out': capped_aged_out,
            'Children_Aged_Out_Difference': capped_aged_out - uncapped_aged_out,
            'Uncapped_India_Backlog': uncapped_india_backlog,
            'Capped_India_Backlog': capped_india_backlog,
            'Uncapped_China_Backlog': uncapped_china_backlog,
            'Capped_China_Backlog': capped_china_backlog,
        })


    # Create DataFrame
    df = pd.DataFrame(comparison_data)


    # Save detailed report
    report_path = output_dir / "conversion_comparison_report.csv"
    df.to_csv(report_path, index=False)
    print(f"Detailed conversion comparison saved to: {report_path}")


    # Print summary statistics for key periods
    print("\n" + "="*80)
    print("CONVERSION COMPARISON ANALYSIS")
    print("="*80)


    # Early years (2026-2040)
    early_years = df[df['Year'].between(2026, 2040)]
    print(f"\nEARLY YEARS (2026-2040):")
    print(f"  Average Uncapped Total Conversions: {early_years['Uncapped_Total_Conversions'].mean():.1f}")
    print(f"  Average Capped Total Conversions: {early_years['Capped_Total_Conversions'].mean():.1f}")
    print(f"  Average Conversion Ratio (Uncapped/Capped): {early_years['Conversion_Ratio'].mean():.2f}")
    print(f"  Average Children Aged Out Difference (Capped - Uncapped): {early_years['Children_Aged_Out_Difference'].mean():.1f}")


    # Late years (2080-2100)
    late_years = df[df['Year'].between(2080, 2100)]
    if not late_years.empty:
        print(f"\nLATE YEARS (2080-2100):")
        print(f"  Average Uncapped Total Conversions: {late_years['Uncapped_Total_Conversions'].mean():.1f}")
        print(f"  Average Capped Total Conversions: {late_years['Capped_Total_Conversions'].mean():.1f}")
        print(f"  Average Conversion Ratio (Uncapped/Capped): {late_years['Conversion_Ratio'].mean():.2f}")
        print(f"  Average Children Aged Out Difference (Capped - Uncapped): {late_years['Children_Aged_Out_Difference'].mean():.1f}")


        # Key diagnostic
        if late_years['Children_Aged_Out_Difference'].mean() < 0:
            print(f"\n  ❌ PROBLEM DETECTED: In late years, uncapped has MORE children aging out than capped!")
            print(f"     This suggests spillover is equalizing scenarios over time.")


        if late_years['Conversion_Ratio'].mean() < 1.2:
            print(f"\n  ❌ SPILLOVER ISSUE: Conversion ratio is too close to 1.0 in late years!")
            print(f"     Capped scenario should consistently process 60-80% of uncapped conversions.")


    # Check for trend reversal in child age-outs
    child_diff_trend = df['Children_Aged_Out_Difference'].rolling(window=10).mean()
    if len(child_diff_trend) > 20:
        early_trend = child_diff_trend.iloc[10:20].mean()
        late_trend = child_diff_trend.iloc[-10:].mean()


        if early_trend > 0 and late_trend < 0:
            print(f"\n  ❌ TREND REVERSAL DETECTED:")
            print(f"     Early years: Capped has {early_trend:.1f} more children aging out (CORRECT)")
            print(f"     Late years: Uncapped has {abs(late_trend):.1f} more children aging out (INCORRECT)")


    print("="*80)


    return df



def get_top_backlogs(backlog_analysis: BacklogAnalysis, top_n: int = 3):
    """Return top N countries by backlog."""
    return sorted(
        backlog_analysis.backlog_by_country.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]



def get_top_eb_category_backlogs(backlog_analysis: BacklogAnalysis, category: EBCategory, top_n: int = 3):
    """Return top N countries by backlog for a specific EB category."""
    category_backlogs = []
    for (cat, nationality), backlog in backlog_analysis.backlog_by_category_nationality.items():
        if cat == category:
            category_backlogs.append((nationality, backlog))


    return sorted(category_backlogs, key=lambda x: x[1], reverse=True)[:top_n]



def print_eb_category_comparison_summary(states_uncapped, states_capped, backlog_uncapped, backlog_capped):
    """Print comprehensive comparison summary with EB category analysis."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EB CATEGORY ANALYSIS SUMMARY")
    print("="*80)


    final_uncapped = states_uncapped[-1]
    final_capped = states_capped[-1]


    # Basic workforce metrics
    print(f"Final workforce size:")
    print(f"  Uncapped: {final_uncapped.total_workers:,}")
    print(f"  Capped: {final_capped.total_workers:,}")
    print(f"  Difference: {final_capped.total_workers - final_uncapped.total_workers:,}\n")


    print(f"Final H-1B share:")
    print(f"  Uncapped: {final_uncapped.h1b_share:.3%}")
    print(f"  Capped: {final_capped.h1b_share:.3%}")
    print(f"  Difference: {final_capped.h1b_share - final_uncapped.h1b_share:.3%}\n")


    print(f"Final average wages:")
    print(f"  Uncapped: ${final_uncapped.avg_wage_total:,.0f}")
    print(f"  Capped: ${final_capped.avg_wage_total:,.0f}")
    print(f"  Difference: ${final_capped.avg_wage_total - final_uncapped.avg_wage_total:,.0f}\n")


    # EB Category conversion analysis
    print("TOTAL CONVERSIONS BY EB CATEGORY:")
    total_conversions_uncapped = {cat: sum(s.converted_by_eb_category.get(cat, 0) for s in states_uncapped[1:]) 
                                   for cat in EBCategory}
    total_conversions_capped = {cat: sum(s.converted_by_eb_category.get(cat, 0) for s in states_capped[1:]) 
                                 for cat in EBCategory}


    for category in [EBCategory.EB1, EBCategory.EB2, EBCategory.EB3]:  # Skip EB4/EB5 (usually 0)
        uncapped_conv = total_conversions_uncapped[category]
        capped_conv = total_conversions_capped[category]
        diff = capped_conv - uncapped_conv
        print(f"  {category.value}: Uncapped={uncapped_conv:,}, Capped={capped_conv:,}, Diff={diff:,}")


    total_uncapped = sum(total_conversions_uncapped.values())
    total_capped = sum(total_conversions_capped.values())
    print(f"  TOTAL: Uncapped={total_uncapped:,}, Capped={total_capped:,}, Diff={total_capped - total_uncapped:,}\n")


    # EB Category backlog analysis
    print("FINAL BACKLOGS BY EB CATEGORY:")
    for category in [EBCategory.EB1, EBCategory.EB2, EBCategory.EB3]:
        uncapped_backlog = backlog_uncapped.backlog_by_eb_category.get(category, 0)
        capped_backlog = backlog_capped.backlog_by_eb_category.get(category, 0)
        diff = capped_backlog - uncapped_backlog
        print(f"  {category.value}: Uncapped={uncapped_backlog:,}, Capped={capped_backlog:,}, Diff={diff:,}")


    print(f"\nCRITICAL EB-2 BACKLOG BREAKDOWN (Advanced Degree Professionals):")
    print(f"  Format: Principals only | With families (principals + spouses + children)")
    for nationality in ['India', 'China', 'Other']:
        # Principals only
        uncapped_eb2 = backlog_uncapped.backlog_by_category_nationality.get((EBCategory.EB2, nationality), 0)
        capped_eb2 = backlog_capped.backlog_by_category_nationality.get((EBCategory.EB2, nationality), 0)
        diff = capped_eb2 - uncapped_eb2
        
        # Family-adjusted (principals + spouses + children)
        uncapped_eb2_family = backlog_uncapped.family_adjusted_backlog_by_category_nationality.get((EBCategory.EB2, nationality), 0)
        capped_eb2_family = backlog_capped.family_adjusted_backlog_by_category_nationality.get((EBCategory.EB2, nationality), 0)
        diff_family = capped_eb2_family - uncapped_eb2_family
        
        print(f"  {nationality}:")
        print(f"    Principals: Uncapped={uncapped_eb2:,}, Capped={capped_eb2:,}, Additional={diff:,}")
        print(f"    With families: Uncapped={uncapped_eb2_family:,}, Capped={capped_eb2_family:,}, Additional={diff_family:,}")


    print(f"\nEB-3 BACKLOG BREAKDOWN (Skilled Workers):")
    print(f"  Format: Principals only | With families (principals + spouses + children)")
    for nationality in ['India', 'China', 'Other']:
        # Principals only
        uncapped_eb3 = backlog_uncapped.backlog_by_category_nationality.get((EBCategory.EB3, nationality), 0)
        capped_eb3 = backlog_capped.backlog_by_category_nationality.get((EBCategory.EB3, nationality), 0)
        diff = capped_eb3 - uncapped_eb3
        
        # Family-adjusted (principals + spouses + children)
        uncapped_eb3_family = backlog_uncapped.family_adjusted_backlog_by_category_nationality.get((EBCategory.EB3, nationality), 0)
        capped_eb3_family = backlog_capped.family_adjusted_backlog_by_category_nationality.get((EBCategory.EB3, nationality), 0)
        diff_family = capped_eb3_family - uncapped_eb3_family
        
        print(f"  {nationality}:")
        print(f"    Principals: Uncapped={uncapped_eb3:,}, Capped={capped_eb3:,}, Additional={diff:,}")
        print(f"    With families: Uncapped={uncapped_eb3_family:,}, Capped={capped_eb3_family:,}, Additional={diff_family:,}")
    
    # Total backlog summary (principals vs families)
    print(f"\nTOTAL BACKLOG SUMMARY:")
    print(f"  Uncapped scenario:")
    print(f"    Principals only: {backlog_uncapped.total_backlog:,}")
    print(f"    With families: {backlog_uncapped.total_family_adjusted_backlog:,}")
    print(f"  Capped scenario:")
    print(f"    Principals only: {backlog_capped.total_backlog:,}")
    print(f"    With families: {backlog_capped.total_family_adjusted_backlog:,}")
    print(f"  Additional people in backlog due to per-country caps:")
    print(f"    Principals: {backlog_capped.total_backlog - backlog_uncapped.total_backlog:,}")
    print(f"    With families: {backlog_capped.total_family_adjusted_backlog - backlog_uncapped.total_family_adjusted_backlog:,}")



    # Child impact analysis
    print(f"\nCHILD AGE-OUT IMPACT:")
    print(f"Total children aged out:")
    print(f"  Uncapped: {final_uncapped.cumulative_children_aged_out:,}")
    print(f"  Capped: {final_capped.cumulative_children_aged_out:,}")
    additional_aged_out = final_capped.cumulative_children_aged_out - final_uncapped.cumulative_children_aged_out
    print(f"  Additional children aged out due to caps: {additional_aged_out:,}")


    if final_uncapped.cumulative_children_aged_out > 0:
        increase_pct = (additional_aged_out / final_uncapped.cumulative_children_aged_out) * 100
        print(f"  Percentage increase due to caps: {increase_pct:.1f}%")


    # Validation of conversion differences
    print(f"\nPER-COUNTRY CAP VALIDATION:")
    total_backlog_uncapped = backlog_uncapped.total_backlog
    total_backlog_capped = backlog_capped.total_backlog
    backlog_diff = abs(total_backlog_uncapped - total_backlog_capped)
    conversions_diff = abs(total_uncapped - total_capped)


    # Large conversion differences are EXPECTED and CORRECT
    print(f"  Total conversions difference: {conversions_diff:,}")
    print(f"  Total backlog difference: {backlog_diff:,}")
    print(f"  Conversion difference validation: {'✅ EXPECTED - Caps working correctly' if conversions_diff > 1000 else '⚠️ WARNING - Difference too small, caps may not be working'}")


    # Additional validation
    if conversions_diff > 1000:
        print(f"  ✅ Per-country caps are functioning as designed")
        print(f"     Uncapped processed {total_uncapped:,} conversions")
        print(f"     Capped processed {total_capped:,} conversions")
        print(f"     Difference: {conversions_diff:,} fewer conversions due to per-country limits")
    else:
        print(f"  ⚠️ WARNING: Expected large conversion difference, got only {conversions_diff}")


    # Key insight
    print(f"\n🔍 KEY INSIGHT:")
    if additional_aged_out > 0:
        print(f"   Per-country caps cause {additional_aged_out:,} more children to age out")
        print(f"   This is due to longer wait times in EB-2 and EB-3 categories for Indian/Chinese families")


        # Calculate most impacted EB category
        eb2_impact = sum(backlog_capped.backlog_by_category_nationality.get((EBCategory.EB2, nat), 0) 
                        for nat in ['India', 'China']) - \
                     sum(backlog_uncapped.backlog_by_category_nationality.get((EBCategory.EB2, nat), 0) 
                        for nat in ['India', 'China'])
        eb3_impact = sum(backlog_capped.backlog_by_category_nationality.get((EBCategory.EB3, nat), 0) 
                        for nat in ['India', 'China']) - \
                     sum(backlog_uncapped.backlog_by_category_nationality.get((EBCategory.EB3, nat), 0) 
                        for nat in ['India', 'China'])


        most_impacted = "EB-2" if eb2_impact > eb3_impact else "EB-3"
        print(f"   Most impacted category: {most_impacted} (where most H-1B holders convert)")
    else:
        print(f"   ⚠️ WARNING: Expected more children to age out in capped scenario - check simulation logic")


    print("="*80)



def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Workforce Growth Simulation - Immigration Policy Analysis with EB Categories",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )


    parser.add_argument('--initial-workers', type=int, required=True, help='Initial workforce size')
    parser.add_argument('--years', type=int, default=20, help='Number of years to simulate')
    parser.add_argument('--country-cap', action='store_true', help='Enable per-country caps (7% within each EB category)')
    parser.add_argument('--compare', action='store_true', help='Compare capped vs uncapped scenarios')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--start-year', type=int, default=2025, help='Starting year')
    parser.add_argument('--output', type=str, default='data/simulation_results.csv', help='Output file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')


    args = parser.parse_args()


    # SCRIPT TIMING START
    script_start_time = time.perf_counter()


    # Setup logging with font suppression
    setup_logging(args.debug)


    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)


    config = create_config_from_args(args)


    print("\n" + "="*80)
    print("WORKFORCE GROWTH SIMULATION WITH EB-1 THROUGH EB-5 CATEGORIES")
    print("="*80)
    print(f"Initial workforce: {config.initial_workers:,}")
    print(f"Simulation years: {config.years}")
    print(f"Per-country cap: {'ENABLED (7% within each EB category)' if config.country_cap_enabled else 'DISABLED'}")
    print(f"Random seed: {config.seed or 'None'}")
    print(f"EB Categories: EB-1 (Priority Workers), EB-2 (Advanced Degree), EB-3 (Skilled Workers)")
    print(f"H-1B Conversion: 5% EB-1, 70% EB-2, 25% EB-3")
    print("="*80)


    try:
        if args.compare:
            run_comparative_analysis(config)
        else:
            run_single_simulation(config)


        logger.info("EB category simulation completed successfully!")


    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


    # SCRIPT TIMING END
    total_script_time = time.perf_counter() - script_start_time
    print(f"\n🕐 Total script runtime: {total_script_time:.3f} seconds")



if __name__ == '__main__':
    main()
