"""
Trip Generator for bike-sharing simulation.

Loads synthetic trip data and provides trips in chronological order
for simulation episodes.
"""

import pandas as pd
from datetime import datetime, time


class TripGenerator:
    """Generates trips from synthetic data for simulation."""
    
    def __init__(self, trips_file):
        """
        Initialize trip generator.
        
        Args:
            trips_file: Path to trips CSV file
        """
        self.trips_file = trips_file
        self.trips_df = None
        self.load_trips()
    
    def load_trips(self):
        """Load trips from CSV file."""
        print(f"Loading trips from {self.trips_file}...")
        self.trips_df = pd.read_csv(self.trips_file)
        
        # Convert timestamps
        self.trips_df['departure_time'] = pd.to_datetime(self.trips_df['departure_time'])
        self.trips_df['arrival_time'] = pd.to_datetime(self.trips_df['arrival_time'])
        
        # Extract useful time features
        self.trips_df['departure_hour'] = self.trips_df['departure_time'].dt.hour
        self.trips_df['departure_minute'] = self.trips_df['departure_time'].dt.minute
        
        # Sort by departure time
        self.trips_df = self.trips_df.sort_values('departure_time').reset_index(drop=True)
        
        print(f"  Loaded {len(self.trips_df):,} trips")
        print(f"  Date range: {self.trips_df['date'].min()} to {self.trips_df['date'].max()}")
    
    def get_days(self):
        """
        Get list of unique days in the dataset.
        
        Returns:
            list: Sorted list of unique dates
        """
        return sorted(self.trips_df['date'].unique())
    
    def get_trips_for_day(self, day, start_hour=7, end_hour=11):
        """
        Get all trips for a specific day within the planning horizon.
        
        Args:
            day: Date string (YYYY-MM-DD) or day number
            start_hour: Start of planning horizon (default: 7am)
            end_hour: End of planning horizon (default: 11am)
        
        Returns:
            pd.DataFrame: Filtered trips for this day
        """
        # Handle day number or date string
        if isinstance(day, int):
            days = self.get_days()
            if day < 1 or day > len(days):
                raise ValueError(f"Day {day} out of range (1-{len(days)})")
            date_str = days[day - 1]
        else:
            date_str = day
        
        # Filter by date
        day_trips = self.trips_df[self.trips_df['date'] == date_str].copy()
        
        # Filter by planning horizon (7am-11am by default)
        day_trips = day_trips[
            (day_trips['departure_hour'] >= start_hour) &
            (day_trips['departure_hour'] < end_hour)
        ].copy()
        
        # Sort by departure time
        day_trips = day_trips.sort_values('departure_time').reset_index(drop=True)
        
        return day_trips
    
    def get_episode_trips(self, day):
        """
        Get trips for a single episode (one day, 7am-11am).
        
        This is the main method used during simulation.
        
        Args:
            day: Day number or date string
        
        Returns:
            list: List of trip dictionaries
        """
        trips_df = self.get_trips_for_day(day)
        
        # Convert to list of dictionaries for easier processing
        trips = []
        for _, row in trips_df.iterrows():
            trip = {
                'trip_id': row['trip_id'],
                'origin_station': row['origin_station'],
                'destination_station': row['destination_station'],
                'departure_time': row['departure_time'],
                'arrival_time': row['arrival_time'],
                'duration_sec': row['duration_sec'],
                'trip_type': row['trip_type']
            }
            trips.append(trip)
        
        return trips
    
    def get_statistics(self):
        """
        Get statistics about the trip dataset.
        
        Returns:
            dict: Dataset statistics
        """
        return {
            'total_trips': len(self.trips_df),
            'num_days': self.trips_df['date'].nunique(),
            'avg_trips_per_day': len(self.trips_df) / self.trips_df['date'].nunique(),
            'date_range': (self.trips_df['date'].min(), self.trips_df['date'].max()),
            'trip_types': self.trips_df['trip_type'].value_counts().to_dict(),
            'stations_used': {
                'origins': self.trips_df['origin_station'].nunique(),
                'destinations': self.trips_df['destination_station'].nunique()
            }
        }
    
    def get_hourly_demand(self, day):
        """
        Get hourly demand pattern for a specific day.
        
        Args:
            day: Day number or date string
        
        Returns:
            dict: Hourly trip counts
        """
        trips_df = self.get_trips_for_day(day, start_hour=0, end_hour=24)
        hourly = trips_df.groupby('departure_hour').size().to_dict()
        return hourly
    
    def __len__(self):
        """Return total number of trips."""
        return len(self.trips_df)
    
    def __repr__(self):
        return (f"TripGenerator(trips={len(self.trips_df):,}, "
                f"days={self.trips_df['date'].nunique()})")
