"""
Train hourly demand prediction model using weather and trip data.

Based on Reference 22 methodology:
- Features: temperature, humidity, hour, weekday
- Algorithm: Linear Regression (OLS)
- Target: Total hourly rentals (system-wide)
- Training: Real BIXI trips + weather (2019 May-Sep weekdays)

The trained model will be used to predict hourly demand from synthetic weather.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def load_bixi_trips():
    """Load BIXI trip data for 2019 May-September weekdays"""
    data_path = Path("../data/bixi/bixi_trips_2019_may_sep_weekdays.csv")
    
    print("Loading BIXI trip data...")
    print(f"  File: {data_path}")
    
    # Load with specific columns to save memory
    df = pd.read_csv(
        data_path,
        usecols=['start_date'],
        dtype={'start_date': str}
    )
    
    print(f"  ✓ Loaded {len(df):,} trips")
    
    # Parse datetime
    df['DateTime'] = pd.to_datetime(df['start_date'])
    df['Date'] = df['DateTime'].dt.date
    df['Hour'] = df['DateTime'].dt.hour
    
    print(f"  Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    
    return df

def aggregate_hourly_demand(df_trips):
    """Aggregate trips to hourly counts"""
    print("\nAggregating trips to hourly demand...")
    
    # Group by date and hour, count trips
    hourly_demand = df_trips.groupby(['Date', 'Hour']).size().reset_index(name='trip_count')
    
    print(f"  ✓ Created {len(hourly_demand):,} hourly observations")
    print(f"  Hourly demand statistics:")
    print(f"    Min:    {hourly_demand['trip_count'].min():,} trips/hour")
    print(f"    Max:    {hourly_demand['trip_count'].max():,} trips/hour")
    print(f"    Mean:   {hourly_demand['trip_count'].mean():.0f} trips/hour")
    print(f"    Median: {hourly_demand['trip_count'].median():.0f} trips/hour")
    
    return hourly_demand

def load_weather_data():
    """Load weather data for 2019 May-September weekdays"""
    data_path = Path("../data/weather/montreal_weather_2019_may_sep_weekdays.csv")
    
    print("\nLoading weather data...")
    print(f"  File: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Parse datetime
    df['DateTime'] = pd.to_datetime(df['Date/Time (LST)'])
    df['Date'] = df['DateTime'].dt.date
    df['Hour'] = df['DateTime'].dt.hour
    
    # Keep only needed columns
    weather = df[['Date', 'Hour', 'Temp (°C)', 'Rel Hum (%)']].copy()
    weather.columns = ['Date', 'Hour', 'Temperature', 'Humidity']
    
    print(f"  ✓ Loaded {len(weather):,} hourly weather records")
    print(f"  Temperature range: {weather['Temperature'].min():.1f}°C to {weather['Temperature'].max():.1f}°C")
    print(f"  Humidity range: {weather['Humidity'].min():.1f}% to {weather['Humidity'].max():.1f}%")
    
    return weather

def merge_data(hourly_demand, weather):
    """Merge trip demand with weather data"""
    print("\nMerging demand and weather data...")
    
    # Merge on Date and Hour
    merged = hourly_demand.merge(weather, on=['Date', 'Hour'], how='inner')
    
    print(f"  ✓ Matched {len(merged):,} hour-records")
    
    # Add weekday feature (0=Monday, 6=Sunday)
    merged['Date'] = pd.to_datetime(merged['Date'])
    merged['Weekday'] = merged['Date'].dt.dayofweek
    
    # Check for missing values
    missing = merged.isnull().sum()
    if missing.any():
        print(f"  ⚠ Missing values detected:")
        print(missing[missing > 0])
        # Drop rows with missing values
        merged = merged.dropna()
        print(f"  After dropping NaN: {len(merged):,} records")
    
    return merged

def train_model(data):
    """Train linear regression model"""
    print("\nTraining linear regression model...")
    
    # Prepare features and target
    feature_cols = ['Temperature', 'Humidity', 'Hour', 'Weekday']
    X = data[feature_cols].values
    y = data['trip_count'].values
    
    print(f"  Features: {feature_cols}")
    print(f"  Training samples: {len(X):,}")
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"  ✓ Model trained")
    
    return model, feature_cols

def evaluate_model(model, X, y, feature_names):
    """Evaluate model performance"""
    print("\nEvaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Coefficients
    print(f"\n  Model Coefficients:")
    print(f"    Intercept:    {model.intercept_:+10.2f}")
    for name, coef in zip(feature_names, model.coef_):
        print(f"    {name:12s}: {coef:+10.2f}")
    
    print(f"\n  Performance Metrics:")
    print(f"    R² Score:     {r2:.4f}")
    print(f"    RMSE:         {rmse:.1f} trips/hour")
    print(f"    MAE:          {mae:.1f} trips/hour")
    
    # Calculate percentage error
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    print(f"    MAPE:         {mape:.1f}%")
    
    metrics = {
        'r2_score': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'intercept': float(model.intercept_),
        'coefficients': {name: float(coef) for name, coef in zip(feature_names, model.coef_)}
    }
    
    return y_pred, metrics

def plot_validation(y_true, y_pred, metrics, output_dir):
    """Create validation plots"""
    print("\nGenerating validation plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Actual vs Predicted scatter
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Hourly Demand (trips)', fontsize=12)
    ax1.set_ylabel('Predicted Hourly Demand (trips)', fontsize=12)
    ax1.set_title(f'Actual vs Predicted Demand\nR² = {metrics["r2_score"]:.4f}', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Predicted Hourly Demand (trips)', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.set_title(f'Residual Plot\nRMSE = {metrics["rmse"]:.1f} trips', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "demand_model_validation.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved validation plot to {output_file}")
    plt.close()

def save_model(model, feature_names, metrics, output_dir):
    """Save trained model and metadata"""
    
    # Save model (pickle)
    model_file = output_dir / "demand_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved model to {model_file}")
    
    # Save metadata (JSON)
    metadata = {
        'features': feature_names,
        'metrics': metrics,
        'model_type': 'LinearRegression',
        'training_data': 'BIXI Montreal 2019 May-Sep weekdays',
        'usage': 'Predict system-wide hourly demand from weather features'
    }
    
    metadata_file = output_dir / "demand_model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")
    
    return model_file, metadata_file

def main():
    """Main training pipeline"""
    print("="*70)
    print("DEMAND MODEL TRAINING")
    print("="*70)
    print("\nObjective: Train regression model (weather → hourly demand)")
    print("Reference: Paper Reference 22 methodology")
    print("Algorithm: Linear Regression (OLS)\n")
    
    # Create output directory
    output_dir = Path("../data/synthetic/fitted_parameters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load BIXI trips
    df_trips = load_bixi_trips()
    
    # Step 2: Aggregate to hourly demand
    hourly_demand = aggregate_hourly_demand(df_trips)
    
    # Step 3: Load weather data
    weather = load_weather_data()
    
    # Step 4: Merge datasets
    data = merge_data(hourly_demand, weather)
    
    # Step 5: Train model
    model, feature_names = train_model(data)
    
    # Step 6: Evaluate model
    X = data[feature_names].values
    y = data['trip_count'].values
    y_pred, metrics = evaluate_model(model, X, y, feature_names)
    
    # Step 7: Plot validation
    plot_validation(y, y_pred, metrics, output_dir)
    
    # Step 8: Save model
    model_file, metadata_file = save_model(model, feature_names, metrics, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    print(f"\n✓ Model Performance:")
    print(f"  R² Score:  {metrics['r2_score']:.4f} {'✓ Good' if metrics['r2_score'] > 0.6 else '⚠ Could be better'}")
    print(f"  RMSE:      {metrics['rmse']:.1f} trips/hour")
    print(f"  MAE:       {metrics['mae']:.1f} trips/hour")
    print(f"  MAPE:      {metrics['mape']:.1f}%")
    
    print(f"\n✓ Key Insights:")
    print(f"  Temperature effect: {metrics['coefficients']['Temperature']:+.1f} trips per °C")
    print(f"  Humidity effect:    {metrics['coefficients']['Humidity']:+.1f} trips per %")
    print(f"  Hour effect:        {metrics['coefficients']['Hour']:+.1f} trips per hour")
    print(f"  Weekday effect:     {metrics['coefficients']['Weekday']:+.1f} trips per day")
    
    print(f"\n✓ Files saved:")
    print(f"  Model:      {model_file}")
    print(f"  Metadata:   {metadata_file}")
    print(f"  Plot:       {output_dir / 'demand_model_validation.png'}")
    
    print("\n✅ Ready for synthetic trip generation!")

if __name__ == "__main__":
    main()
