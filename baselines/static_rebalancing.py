"""
Static Rebalancing for Initial Inventory Computation.

Based on Reference 21, Section 5.2, Equations (28)-(32).

The static rebalancing problem finds optimal initial station inventories
by minimizing expected lost demand over the planning horizon, using
average demand patterns from the training set.

This solution provides the initial state for ALL methods (DQN, MIP, baselines).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from pulp import *


def compute_average_demands(trips_file, network_file, start_hour=7, end_hour=11, period_minutes=30):
    """
    Compute average rental and return demands per station per time period.
    
    Based on Reference 21, lines 1816-1824:
    "We calculate the average rental and return demands for each instance 
    over the training set at each station of each time-period."
    
    Args:
        trips_file: Path to training trips CSV
        network_file: Path to station network JSON
        start_hour: Planning horizon start (default: 7am)
        end_hour: Planning horizon end (default: 11am)
        period_minutes: Time period length in minutes (default: 30)
    
    Returns:
        dict: {station_id: {period: {'rentals': avg, 'returns': avg}}}
    """
    print(f"\n{'='*70}")
    print("COMPUTING AVERAGE DEMANDS FROM TRAINING DATA")
    print(f"{'='*70}")
    
    # Load network to get all station IDs
    with open(network_file) as f:
        network = json.load(f)
    station_ids = [s['id'] for s in network['stations']]
    
    # Load trips
    print(f"Loading trips from {Path(trips_file).name}...")
    trips_df = pd.read_csv(trips_file)
    trips_df['departure_time'] = pd.to_datetime(trips_df['departure_time'])
    trips_df['arrival_time'] = pd.to_datetime(trips_df['arrival_time'])
    
    # Add time features
    trips_df['departure_hour'] = trips_df['departure_time'].dt.hour
    trips_df['departure_minute'] = trips_df['departure_time'].dt.minute
    trips_df['arrival_hour'] = trips_df['arrival_time'].dt.hour
    trips_df['arrival_minute'] = trips_df['arrival_time'].dt.minute
    trips_df['date'] = trips_df['departure_time'].dt.date
    
    # Filter for planning horizon
    planning_trips = trips_df[
        (trips_df['departure_hour'] >= start_hour) & 
        (trips_df['departure_hour'] < end_hour)
    ].copy()
    
    print(f"  Total trips in dataset: {len(trips_df):,}")
    print(f"  Trips in planning horizon ({start_hour}am-{end_hour}am): {len(planning_trips):,}")
    print(f"  Number of days: {trips_df['date'].nunique()}")
    
    # Calculate number of periods
    planning_hours = end_hour - start_hour
    num_periods = (planning_hours * 60) // period_minutes
    
    print(f"\nTime discretization:")
    print(f"  Planning horizon: {planning_hours} hours")
    print(f"  Period length: {period_minutes} minutes")
    print(f"  Number of periods: {num_periods}")
    
    # Assign trips to periods
    def get_period(hour, minute):
        """Convert hour:minute to period index (0-indexed)."""
        minutes_from_start = (hour - start_hour) * 60 + minute
        period = minutes_from_start // period_minutes
        return max(0, min(period, num_periods - 1))
    
    planning_trips['departure_period'] = planning_trips.apply(
        lambda row: get_period(row['departure_hour'], row['departure_minute']),
        axis=1
    )
    planning_trips['arrival_period'] = planning_trips.apply(
        lambda row: get_period(row['arrival_hour'], row['arrival_minute']) 
        if row['arrival_hour'] < end_hour else num_periods - 1,
        axis=1
    )
    
    # Count rentals and returns per station per period per day
    num_days = trips_df['date'].nunique()
    
    # Initialize demand structure
    average_demands = {}
    for station_id in station_ids:
        average_demands[station_id] = {}
        for period in range(num_periods):
            average_demands[station_id][period] = {'rentals': 0.0, 'returns': 0.0}
    
    # Aggregate rentals (origins)
    print("\nAggregating rental demands...")
    rentals = planning_trips.groupby(['origin_station', 'departure_period']).size()
    for (station, period), count in rentals.items():
        if station in station_ids:
            average_demands[station][period]['rentals'] = count / num_days
    
    # Aggregate returns (destinations)
    print("Aggregating return demands...")
    returns = planning_trips.groupby(['destination_station', 'arrival_period']).size()
    for (station, period), count in returns.items():
        if station in station_ids and period < num_periods:
            average_demands[station][period]['returns'] = count / num_days
    
    # Summary statistics
    total_avg_rentals = sum(
        average_demands[s][p]['rentals'] 
        for s in station_ids for p in range(num_periods)
    )
    total_avg_returns = sum(
        average_demands[s][p]['returns'] 
        for s in station_ids for p in range(num_periods)
    )
    
    print(f"\nAverage demands per day:")
    print(f"  Total rentals: {total_avg_rentals:.1f}")
    print(f"  Total returns: {total_avg_returns:.1f}")
    print(f"  Per period: {total_avg_rentals / num_periods:.1f} rentals, {total_avg_returns / num_periods:.1f} returns")
    
    # Show sample demands
    print(f"\nSample average demands (Station 1, all periods):")
    for period in range(min(num_periods, 8)):
        r = average_demands[1][period]['rentals']
        ret = average_demands[1][period]['returns']
        print(f"  Period {period} ({start_hour + period * period_minutes // 60:02d}:{(period * period_minutes) % 60:02d}): "
              f"{r:.1f} rentals, {ret:.1f} returns")
    
    return average_demands, num_periods


def solve_static_rebalancing_mip(network, average_demands, num_periods, total_bikes=None, optimize_bike_count=False):
    """
    Solve static rebalancing optimization problem using MIP.
    
    Based on Reference 21, Section 5.2, Equations (28)-(32):
    
    minimize  Σ_s Σ_t (f^+_{s,t} - x^+_{s,t}) + (f^-_{s,t} - x^-_{s,t})
    
    subject to:
        d^{t+1}_s = d^t_s - x^+_{s,t} + x^-_{s,t}   (inventory evolution)
        x^+_{s,t} <= d^t_s                           (rentals limited by inventory)
        x^-_{s,t} <= C_s - d^t_s                     (returns limited by available docks)
        x^+_{s,t} <= f^+_{s,t}                       (rentals limited by demand)
        x^-_{s,t} <= f^-_{s,t}                       (returns limited by demand)
        Σ_s d^1_s = B                                (total bikes constraint)
    
    Args:
        network: Station network dict
        average_demands: Average demand dict from compute_average_demands()
        num_periods: Number of time periods
        total_bikes: Total bikes in system (default: 50% of total capacity)
        optimize_bike_count: If True, treat total_bikes as a decision variable
    
    Returns:
        dict: {station_id: initial_inventory}
    """
    print(f"\n{'='*70}")
    print("SOLVING STATIC REBALANCING OPTIMIZATION")
    print(f"{'='*70}")
    
    stations = network['stations']
    station_ids = [s['id'] for s in stations]
    total_capacity = sum(s['capacity'] for s in stations)
    
    # Calculate total bikes bounds
    if optimize_bike_count:
        # Treat total bikes as decision variable with bounds [30%, 70%] of capacity
        min_bikes = int(total_capacity * 0.30)
        max_bikes = int(total_capacity * 0.70)
        print(f"\n*** OPTIMIZING TOTAL BIKE COUNT ***")
        print(f"  Bike count range: [{min_bikes}, {max_bikes}] (30%-70% of capacity)")
    else:
        # Fixed total bikes
        if total_bikes is None:
            total_bikes = total_capacity // 2
        min_bikes = total_bikes
        max_bikes = total_bikes
    
    print(f"\nProblem size:")
    print(f"  Stations: {len(stations)}")
    print(f"  Time periods: {num_periods}")
    print(f"  Total capacity: {total_capacity}")
    if optimize_bike_count:
        print(f"  Total bikes: DECISION VARIABLE [{min_bikes}, {max_bikes}]")
    else:
        print(f"  Total bikes: {total_bikes} (fixed)")
    
    # Create optimization problem
    prob = LpProblem("StaticRebalancing", LpMinimize)
    
    # Decision variables
    print("\nCreating decision variables...")
    
    # d[s,t] = inventory at station s at beginning of period t
    d = {}
    for s in stations:
        for t in range(num_periods + 1):
            d[s['id'], t] = LpVariable(
                f"d_{s['id']}_{t}",
                lowBound=0,
                upBound=s['capacity'],
                cat='Integer'
            )
    
    # x_plus[s,t] = successful rentals at station s in period t
    # x_minus[s,t] = successful returns at station s in period t
    x_plus = {}
    x_minus = {}
    
    for s in stations:
        for t in range(num_periods):
            sid = s['id']
            f_plus = average_demands[sid][t]['rentals']
            f_minus = average_demands[sid][t]['returns']
            
            # Rentals bounded by demand
            x_plus[sid, t] = LpVariable(
                f"x_plus_{sid}_{t}",
                lowBound=0,
                upBound=f_plus
            )
            
            # Returns bounded by demand
            x_minus[sid, t] = LpVariable(
                f"x_minus_{sid}_{t}",
                lowBound=0,
                upBound=f_minus
            )
    
    print(f"  Created {len(d)} inventory variables")
    print(f"  Created {len(x_plus)} rental variables")
    print(f"  Created {len(x_minus)} return variables")
    
    # Objective function: Minimize lost demand
    print("\nSetting objective function...")
    
    lost_rentals = lpSum([
        average_demands[s['id']][t]['rentals'] - x_plus[s['id'], t]
        for s in stations for t in range(num_periods)
    ])
    
    lost_returns = lpSum([
        average_demands[s['id']][t]['returns'] - x_minus[s['id'], t]
        for s in stations for t in range(num_periods)
    ])
    
    prob += lost_rentals + lost_returns, "TotalLostDemand"
    
    # Constraints
    print("Adding constraints...")
    
    # Constraint 1: Total bikes in system
    total_initial_bikes = lpSum([d[s['id'], 0] for s in stations])
    if optimize_bike_count:
        # Treat as decision variable with bounds
        prob += total_initial_bikes >= min_bikes, "MinBikes"
        prob += total_initial_bikes <= max_bikes, "MaxBikes"
        print(f"  Total bikes: decision variable in [{min_bikes}, {max_bikes}]")
    else:
        prob += total_initial_bikes == total_bikes, "TotalBikes"
        print(f"  Total bikes: fixed at {total_bikes}")
    
    # Constraint 2: Inventory evolution (d^{t+1}_s = d^t_s - x^{+,t}_s + x^{-,t}_s)
    constraint_count = 0
    for s in stations:
        for t in range(num_periods):
            sid = s['id']
            prob += (
                d[sid, t + 1] == d[sid, t] - x_plus[sid, t] + x_minus[sid, t],
                f"InventoryEvolution_{sid}_{t}"
            )
            constraint_count += 1
    print(f"  Added {constraint_count} inventory evolution constraints")
    
    # Constraint 3: Rentals limited by available bikes
    constraint_count = 0
    for s in stations:
        for t in range(num_periods):
            sid = s['id']
            prob += x_plus[sid, t] <= d[sid, t], f"RentalLimit_{sid}_{t}"
            constraint_count += 1
    print(f"  Added {constraint_count} rental limit constraints")
    
    # Constraint 4: Returns limited by available docks
    constraint_count = 0
    for s in stations:
        for t in range(num_periods):
            sid = s['id']
            available_docks = s['capacity'] - d[sid, t]
            prob += x_minus[sid, t] <= available_docks, f"ReturnLimit_{sid}_{t}"
            constraint_count += 1
    print(f"  Added {constraint_count} return limit constraints")
    
    # Solve the problem
    print(f"\n{'='*70}")
    print("SOLVING MIP...")
    print(f"{'='*70}")
    
    solver = PULP_CBC_CMD(msg=1, timeLimit=300)  # 5 minute time limit
    prob.solve(solver)
    
    # Check solution status
    status = LpStatus[prob.status]
    print(f"\nSolution Status: {status}")
    
    if status != 'Optimal' and status != 'Feasible':
        raise RuntimeError(f"MIP solver failed with status: {status}")
    
    # Extract results
    initial_inventory = {}
    for s in stations:
        initial_inventory[s['id']] = int(value(d[s['id'], 0]))
    
    # Calculate objective value
    obj_value = value(prob.objective)
    total_demand = sum(
        average_demands[s['id']][t]['rentals'] + average_demands[s['id']][t]['returns']
        for s in stations for t in range(num_periods)
    )
    lost_demand_pct = (obj_value / total_demand) * 100 if total_demand > 0 else 0
    
    print(f"\nObjective Value (Lost Demand): {obj_value:.2f}")
    print(f"Total Average Demand: {total_demand:.2f}")
    print(f"Lost Demand Rate: {lost_demand_pct:.2f}%")
    
    # Inventory statistics
    inventories = list(initial_inventory.values())
    print(f"\nInitial Inventory Statistics:")
    print(f"  Total bikes: {sum(inventories)}")
    print(f"  Mean: {np.mean(inventories):.1f}")
    print(f"  Std: {np.std(inventories):.1f}")
    print(f"  Min: {min(inventories)}")
    print(f"  Max: {max(inventories)}")
    
    # Show sample inventories
    print(f"\nSample Initial Inventories:")
    for s in stations[:10]:
        inv = initial_inventory[s['id']]
        occ = inv / s['capacity']
        cc_label = " (city center)" if s['is_city_center'] else ""
        print(f"  Station {s['id']}: {inv}/{s['capacity']} bikes ({occ*100:.1f}%){cc_label}")
    
    return initial_inventory


def compute_and_save_static_inventory(gt_name='GT1', start_hour=7, end_hour=11, 
                                       optimize_bike_count=False, bike_fraction=0.5):
    """
    Complete workflow: compute and save static initial inventory.
    
    Args:
        gt_name: Ground truth name ('GT1' or 'GT2')
        start_hour: Planning horizon start
        end_hour: Planning horizon end
        optimize_bike_count: If True, treat total_bikes as decision variable (overrides bike_fraction)
        bike_fraction: Fraction of total capacity for bikes (default: 0.5 = 50%)
    
    Returns:
        dict: Initial inventory {station_id: bikes}
    """
    print(f"\n{'='*70}")
    print(f"STATIC REBALANCING FOR {gt_name}")
    print(f"{'='*70}")
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    network_file = base_dir / f'{gt_name}_station_network.json'
    trips_file = base_dir / f'{gt_name}_trips_train.csv'
    output_file = base_dir / f'{gt_name}_static_initial_inventory.json'
    
    print(f"\nInput files:")
    print(f"  Network: {network_file}")
    print(f"  Trips: {trips_file}")
    print(f"Output file:")
    print(f"  {output_file}")
    
    # Load network
    with open(network_file) as f:
        network = json.load(f)
    
    # Step 1: Compute average demands
    average_demands, num_periods = compute_average_demands(
        trips_file, network_file, start_hour, end_hour
    )
    
    # Calculate total bikes from fraction if not optimizing
    total_capacity = sum(s['capacity'] for s in network['stations'])
    total_bikes = int(total_capacity * bike_fraction) if not optimize_bike_count else None
    
    # Step 2: Solve optimization
    initial_inventory = solve_static_rebalancing_mip(
        network, average_demands, num_periods, 
        total_bikes=total_bikes, optimize_bike_count=optimize_bike_count
    )
    
    # Step 3: Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    output_data = {
        'gt_name': gt_name,
        'planning_horizon': f'{start_hour}am-{end_hour}am',
        'num_periods': num_periods,
        'total_bikes': sum(initial_inventory.values()),
        'initial_inventory': initial_inventory,
        'metadata': {
            'generated': datetime.now().isoformat(),
            'method': 'static_rebalancing_mip',
            'reference': 'Reference 21, Equations (28)-(32)'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Saved static initial inventory to:")
    print(f"   {output_file}")
    print(f"\n   Total bikes: {sum(initial_inventory.values())}")
    print(f"   Number of stations: {len(initial_inventory)}")
    
    return initial_inventory


if __name__ == "__main__":
    """
    Generate static initial inventories for GT1 and GT2.
    
    Usage:
        python static_rebalancing.py                           # Both GT1 and GT2 with 50% bikes
        python static_rebalancing.py --gt GT1                  # GT1 only with 50% bikes
        python static_rebalancing.py --gt GT1 --bike-fraction 0.3   # GT1 with 30% bikes
        python static_rebalancing.py --gt GT1 --optimize       # GT1 with MIP-optimized bike count
    """
    import sys
    import argparse
    
    # Check if pulp is installed
    try:
        from pulp import *
    except ImportError:
        print("\n❌ Error: PuLP is required but not installed.")
        print("\nInstall it with:")
        print("  pip install pulp")
        sys.exit(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Static Rebalancing MIP Solver')
    parser.add_argument('--gt', type=str, choices=['GT0', 'GT1', 'GT2', 'both'], default='both',
                        help='Ground truth dataset (GT1, GT2, or both)')
    parser.add_argument('--bike-fraction', type=float, default=0.5,
                        help='Fraction of total capacity for bikes (default: 0.5 = 50%%)')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize bike count using MIP (overrides --bike-fraction)')
    args = parser.parse_args()
    
    # Determine which ground truths to process
    gt_list = ['GT1', 'GT2'] if args.gt == 'both' else [args.gt]
    
    print(f"\n{'='*70}")
    print("STATIC REBALANCING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Ground truths: {gt_list}")
    if args.optimize:
        print(f"Bike count: MIP-OPTIMIZED (30%-70% of capacity)")
    else:
        print(f"Bike count: {args.bike_fraction*100:.0f}% of total capacity")
    
    # Generate for selected ground truths
    for gt_name in gt_list:
        try:
            compute_and_save_static_inventory(
                gt_name, 
                optimize_bike_count=args.optimize,
                bike_fraction=args.bike_fraction
            )
            print(f"\n{'='*70}\n")
        except Exception as e:
            print(f"\n❌ Error processing {gt_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("✅ Static rebalancing complete!")
