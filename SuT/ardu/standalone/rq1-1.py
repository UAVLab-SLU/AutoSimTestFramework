from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
import time

from pymavlink import mavutil

# Connect to the Vehicle
vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)
print("Connected to the vehicle!")

# Define NED waypoints
waypoints = [
    [0, 0, -5],
    [0, 10, -5],
    [10, 10, -5],
    [10, 0, -5],
    [0, 0, -5]
]

# Set home location
vehicle.home_location = vehicle.location.global_frame

# Function to send NED position commands
def send_ned_position_target(north, east, down):
    """
    Sends a message specifying a target location relative to the home location.
    :param north: North position in meters
    :param east: East position in meters
    :param down: Down position in meters (negative for below home altitude)
    """
    msg = vehicle.message_factory.mavlink_position_target_local_ned_t(
        0,  # System ID
        0,  # Component ID
        0,  # Time since system boot in milliseconds
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # MAV frame
        0b0000111111111000,  # Target type mask (all position fields)
        north,  # North position in meters
        east,  # East position in meters
        down,  # Down position in meters (negative for below home altitude)
        0,  # X velocity in meters per second
        0,  # Y velocity in meters per second
        0,  # Z velocity in meters per second
        0,  # Yaw angle in degrees
        0,  # Yaw rate in degrees per second
        0,  # Relative altitude in meters
        0,  # Time in UTC milliseconds since the message was sent
    )
    vehicle.send_message(msg)

# Function to send NED velocity commands
def send_ned_velocity(north, east, down):
    """
    Sends a message specifying the speed components of the vehicle in the NED frame.
    :param north: North velocity in meters per second
    :param east: East velocity in meters per second
    :param down: Down velocity in meters per second (negative for below home altitude)
    """
    msg = vehicle.message_factory.mavlink_set_position_target_local_ned_t(
        0,  # System ID
        0,  # Component ID
        0,  # Time since system boot in milliseconds
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # MAV frame
        0,  # Target type mask (all velocity fields)
        0,  # Ignore flags
        north,  # North velocity in meters per second
        east,  # East velocity in meters per second
        down,  # Down velocity in meters per second (negative for below home altitude)
        0,  # X acceleration in meters per second squared
        0,  # Y acceleration in meters per second squared
        0,  # Z acceleration in meters per second squared
        0,  # Yaw angle in degrees
        0,  # Yaw rate in degrees per second
    )
    vehicle.send_message(msg)

# Arm and takeoff
vehicle.mode = VehicleMode("GUIDED")

# position estimation
vehicle.airspeed = 3

# Set home location
vehicle.home_location = vehicle.location.global_frame
vehicle.home_location.lat = 27.994402
vehicle.home_location.lon = -82.582034

print("Arming the vehicle")

vehicle.armed = True

while not vehicle.armed:
    print("Waiting for arming...")
    time.sleep(1)

print("Taking off!")
vehicle.simple_takeoff(5)

while True:
    print("Altitude:", vehicle.location.global_relative_frame.alt)
    if vehicle.location.global_relative_frame.alt >= 4.5:
        print("Target altitude reached")
        break
    time.sleep(1)

# Fly the square path using NED position commands
for point in waypoints:
    north, east, down = point
    send_ned_position_target(north, east, down)

    while True:
        remaining_distance = vehicle.location.global_relative_frame.distance_to([north, east, down])
        print("Distance to target:", remaining_distance)
        if remaining_distance <= 1:  # Adjust this threshold as needed
            print("Reached target")
            break
        time.sleep(1)

# Land the vehicle
vehicle.mode = VehicleMode("LAND")

while vehicle.armed:
    print("Altitude:", vehicle.location.global_relative_frame.alt)
    time.sleep(1)

print("Disarmed")

# Close the connection
vehicle.close()
