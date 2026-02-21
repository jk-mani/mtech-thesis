"""
Event definitions for continuous-time simulation.

Based on Base Paper Section 4.2 and Algorithm 1.
"""

from datetime import datetime


class Event:
    """Base event class."""
    def __init__(self, time, event_type):
        self.time = time
        self.event_type = event_type
    
    def __lt__(self, other):
        """For priority queue ordering (earlier events first)."""
        return self.time < other.time
    
    def __repr__(self):
        return f"{self.event_type} at {self.time.strftime('%H:%M:%S')}"


class VehicleArrival(Event):
    """
    Vehicle arrives at a station.
    
    This triggers a decision epoch where the vehicle must:
    1. Observe the current state
    2. Select an action (next station, fill level)
    3. Execute rebalancing operation
    4. Travel to next station
    """
    def __init__(self, time, vehicle_id, station_id):
        super().__init__(time, "VehicleArrival")
        self.vehicle_id = vehicle_id
        self.station_id = station_id


class CustomerRental(Event):
    """
    Customer attempts to rent a bike (trip departure).
    
    If successful, a CustomerReturn event is scheduled.
    If unsuccessful, lost rental demand is recorded.
    """
    def __init__(self, time, trip_id, origin_station, destination_station, arrival_time):
        super().__init__(time, "CustomerRental")
        self.trip_id = trip_id
        self.origin_station = origin_station
        self.destination_station = destination_station
        self.arrival_time = arrival_time


class CustomerReturn(Event):
    """
    Customer attempts to return a bike (trip arrival).
    
    If successful, bike is added to station inventory.
    If unsuccessful (station full), lost return demand is recorded.
    """
    def __init__(self, time, trip_id, station_id, rental_station):
        super().__init__(time, "CustomerReturn")
        self.trip_id = trip_id
        self.station_id = station_id
        self.rental_station = rental_station


def create_rental_event(trip_row):
    """
    Create CustomerRental event from trip data.
    
    Args:
        trip_row: Row from trips DataFrame with departure info
    
    Returns:
        CustomerRental event
    """
    return CustomerRental(
        time=trip_row['departure_time'],
        trip_id=trip_row['trip_id'],
        origin_station=trip_row['origin_station'],
        destination_station=trip_row['destination_station'],
        arrival_time=trip_row['arrival_time']
    )


def create_return_event(trip_id, station_id, arrival_time, rental_station):
    """
    Create CustomerReturn event (scheduled when rental succeeds).
    
    Args:
        trip_id: Trip identifier
        station_id: Destination station
        arrival_time: When customer arrives
        rental_station: Where bike was rented
    
    Returns:
        CustomerReturn event
    """
    return CustomerReturn(
        time=arrival_time,
        trip_id=trip_id,
        station_id=station_id,
        rental_station=rental_station
    )


def create_vehicle_arrival_event(vehicle_id, station_id, arrival_time):
    """
    Create VehicleArrival event (scheduled when vehicle departs).
    
    Args:
        vehicle_id: Vehicle identifier
        station_id: Destination station
        arrival_time: When vehicle arrives
    
    Returns:
        VehicleArrival event
    """
    return VehicleArrival(
        time=arrival_time,
        vehicle_id=vehicle_id,
        station_id=station_id
    )
