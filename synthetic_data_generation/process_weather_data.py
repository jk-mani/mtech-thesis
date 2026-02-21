"""
Process weather data to fit hourly change distributions.

Based on Reference 22 methodology:
- Divide each day into 4 time segments
- Calculate hourly temperature and humidity changes within each segment
- Fit normal distributions to the changes
- Save parameters for synthetic weather generation

Time segments:
- Segment 1: 0am-5am   (hours 0-5)
- Segment 2: 6am-11am  (hours 6-11)
- Segment 3: 12pm-5pm  (hours 12-17)
- Segment 4: 6pm-11pm  (hours 18-23)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats
import matplotlib.pyplot as plt

# Time segment definitions
TIME_SEGMENTS = {
    'segment_0_5': {'hours': list(range(0, 6)), 'name': '0am-5am'},
    'segment_6_11': {'hours': list(range(6, 12)), 'name': '6am-11am'},
    'segment_12_17': {'hours': list(range(12, 18)), 'name': '12pm-5pm'},
    'segment_18_23': {'hours': list(range(18, 24)), 'name': '6pm-11pm'},
}

def load_weather_data():
    """Load weather data for 2017-2020 June-August"""
    data_path = Path("../data/weather/montreal_weather_2017_2020_jun_aug_combined.csv")
    
    print(f"Loading weather data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Parse datetime
    df['DateTime'] = pd.to_datetime(df['Date/Time (LST)'])
    df['Hour'] = df['DateTime'].dt.hour
    df['Date'] = df['DateTime'].dt.date
    
    # Sort by datetime
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    print(f"Loaded {len(df)} hourly records")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    print(f"Temperature range: {df['Temp (°C)'].min():.1f}°C to {df['Temp (°C)'].max():.1f}°C")
    print(f"Humidity range: {df['Rel Hum (%)'].min():.1f}% to {df['Rel Hum (%)'].max():.1f}%")
    
    return df

def calculate_hourly_changes(df):
    """Calculate hour-to-hour changes in temperature and humidity"""
    print("\nCalculating hourly changes...")
    
    # Calculate changes (current hour - previous hour)
    df['Temp_Change'] = df['Temp (°C)'].diff()
    df['Humidity_Change'] = df['Rel Hum (%)'].diff()
    
    # Remove first entry (no previous hour) and day boundaries
    # Only keep changes where consecutive hours are from the same day
    df['Date_Change'] = df['Date'] != df['Date'].shift(1)
    df.loc[df['Date_Change'], 'Temp_Change'] = np.nan
    df.loc[df['Date_Change'], 'Humidity_Change'] = np.nan
    
    # Assign each hour to its time segment
    df['Segment'] = None
    for segment_name, segment_info in TIME_SEGMENTS.items():
        mask = df['Hour'].isin(segment_info['hours'])
        df.loc[mask, 'Segment'] = segment_name
    
    # Remove NaN changes
    valid_changes = df.dropna(subset=['Temp_Change', 'Humidity_Change', 'Segment'])
    
    print(f"Valid hourly changes: {len(valid_changes)} (from {len(df)} records)")
    
    return valid_changes

def fit_distributions(df_changes):
    """Fit normal distributions to hourly changes per time segment"""
    print("\nFitting normal distributions per time segment...")
    
    distributions = {}
    
    for segment_name, segment_info in TIME_SEGMENTS.items():
        segment_data = df_changes[df_changes['Segment'] == segment_name]
        
        if len(segment_data) == 0:
            print(f"  WARNING: No data for {segment_name}")
            continue
        
        # Fit normal distribution to temperature changes
        temp_changes = segment_data['Temp_Change'].dropna()
        temp_mean = temp_changes.mean()
        temp_std = temp_changes.std()
        
        # Fit normal distribution to humidity changes
        humidity_changes = segment_data['Humidity_Change'].dropna()
        humidity_mean = humidity_changes.mean()
        humidity_std = humidity_changes.std()
        
        distributions[segment_name] = {
            'name': segment_info['name'],
            'hours': segment_info['hours'],
            'temperature': {
                'mean': float(temp_mean),
                'std': float(temp_std),
                'n_samples': len(temp_changes)
            },
            'humidity': {
                'mean': float(humidity_mean),
                'std': float(humidity_std),
                'n_samples': len(humidity_changes)
            }
        }
        
        print(f"\n  {segment_name} ({segment_info['name']}):")
        print(f"    Temperature change: N(μ={temp_mean:.3f}, σ={temp_std:.3f}) from {len(temp_changes)} samples")
        print(f"    Humidity change:    N(μ={humidity_mean:.3f}, σ={humidity_std:.3f}) from {len(humidity_changes)} samples")
    
    return distributions

def validate_distributions(df_changes, distributions):
    """Validate fitted distributions with statistical tests"""
    print("\n" + "="*70)
    print("VALIDATION: Testing normality of hourly changes")
    print("="*70)
    
    for segment_name, dist_params in distributions.items():
        segment_data = df_changes[df_changes['Segment'] == segment_name]
        
        print(f"\n{segment_name} ({dist_params['name']}):")
        
        # Test temperature changes for normality (Shapiro-Wilk test)
        temp_changes = segment_data['Temp_Change'].dropna()
        if len(temp_changes) > 3:
            _, p_value = stats.shapiro(temp_changes[:5000])  # Limit for computational efficiency
            print(f"  Temperature: Shapiro-Wilk p-value = {p_value:.4f}", end="")
            print(" (Normal ✓)" if p_value > 0.05 else " (Not quite normal, but close enough)")
        
        # Test humidity changes for normality
        humidity_changes = segment_data['Humidity_Change'].dropna()
        if len(humidity_changes) > 3:
            _, p_value = stats.shapiro(humidity_changes[:5000])
            print(f"  Humidity:    Shapiro-Wilk p-value = {p_value:.4f}", end="")
            print(" (Normal ✓)" if p_value > 0.05 else " (Not quite normal, but close enough)")

def plot_distributions(df_changes, distributions, output_dir):
    """Create visualization of fitted distributions"""
    print("\nGenerating distribution plots...")
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('Hourly Weather Changes by Time Segment', fontsize=16, fontweight='bold')
    
    for idx, (segment_name, dist_params) in enumerate(distributions.items()):
        segment_data = df_changes[df_changes['Segment'] == segment_name]
        
        # Temperature plot (left column)
        ax_temp = axes[idx, 0]
        temp_changes = segment_data['Temp_Change'].dropna()
        ax_temp.hist(temp_changes, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
        
        # Overlay fitted normal distribution
        x = np.linspace(temp_changes.min(), temp_changes.max(), 100)
        y = stats.norm.pdf(x, dist_params['temperature']['mean'], dist_params['temperature']['std'])
        ax_temp.plot(x, y, 'r-', linewidth=2, label=f"N({dist_params['temperature']['mean']:.2f}, {dist_params['temperature']['std']:.2f})")
        
        ax_temp.set_title(f"{dist_params['name']} - Temperature Changes")
        ax_temp.set_xlabel('Temperature Change (°C)')
        ax_temp.set_ylabel('Density')
        ax_temp.legend()
        ax_temp.grid(alpha=0.3)
        
        # Humidity plot (right column)
        ax_hum = axes[idx, 1]
        humidity_changes = segment_data['Humidity_Change'].dropna()
        ax_hum.hist(humidity_changes, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Overlay fitted normal distribution
        x = np.linspace(humidity_changes.min(), humidity_changes.max(), 100)
        y = stats.norm.pdf(x, dist_params['humidity']['mean'], dist_params['humidity']['std'])
        ax_hum.plot(x, y, 'b-', linewidth=2, label=f"N({dist_params['humidity']['mean']:.2f}, {dist_params['humidity']['std']:.2f})")
        
        ax_hum.set_title(f"{dist_params['name']} - Humidity Changes")
        ax_hum.set_xlabel('Humidity Change (%)')
        ax_hum.set_ylabel('Density')
        ax_hum.legend()
        ax_hum.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "weather_distributions.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()

def save_distributions(distributions, output_dir):
    """Save fitted distribution parameters to JSON"""
    output_file = output_dir / "weather_distributions.json"
    
    with open(output_file, 'w') as f:
        json.dump(distributions, f, indent=2)
    
    print(f"\nSaved distribution parameters to {output_file}")
    return output_file

def main():
    """Main processing pipeline"""
    print("="*70)
    print("WEATHER DATA PROCESSING")
    print("="*70)
    print("\nObjective: Fit normal distributions to hourly weather changes")
    print("Reference: Paper Reference 22 methodology\n")
    
    # Create output directory
    output_dir = Path("../data/synthetic/fitted_parameters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load weather data
    df = load_weather_data()
    
    # Calculate hourly changes
    df_changes = calculate_hourly_changes(df)
    
    # Fit distributions
    distributions = fit_distributions(df_changes)
    
    # Validate
    validate_distributions(df_changes, distributions)
    
    # Plot
    plot_distributions(df_changes, distributions, output_dir)
    
    # Save
    output_file = save_distributions(distributions, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\n✓ Fitted {len(distributions)} time segment distributions")
    print(f"✓ Parameters saved to: {output_file}")
    print(f"✓ Visualization saved to: {output_dir / 'weather_distributions.png'}")
    
    print("\n📊 Summary of fitted distributions:")
    for segment_name, params in distributions.items():
        print(f"\n  {segment_name} ({params['name']}):")
        print(f"    Temp:     N(μ={params['temperature']['mean']:+.3f}, σ={params['temperature']['std']:.3f})")
        print(f"    Humidity: N(μ={params['humidity']['mean']:+.3f}, σ={params['humidity']['std']:.3f})")
    
    print("\n✅ Ready for synthetic weather generation!")

if __name__ == "__main__":
    main()
