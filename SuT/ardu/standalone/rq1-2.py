from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

# Connect to the Vehicle
vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)
print("Connected to the vehicle!")

waypoints = [
    LocationGlobalRelative(27.994402, -82.582034, 5),
    LocationGlobalRelative(27.994402, -82.583034, 5),
    LocationGlobalRelative(27.995402, -82.583034, 5),
    LocationGlobalRelative(27.995402, -82.582034, 5),
    LocationGlobalRelative(27.994402, -82.582034, 5)
]

wind_vector = [3.5, 3.5, 0]
vehicle.airspeed = 3

# Set home location (uncomment if needed, adjust coordinates)
vehicle.home_location = vehicle.location.global_frame
vehicle.home_location.lat = 27.994402
vehicle.home_location.lon = -82.582034

print("Arming the vehicle")
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True

while not vehicle.armed:
    print("Waiting for arming...")
    time.sleep(1)

print("Taking off!")
vehicle.simple_takeoff(5)

# Wait for the drone to reach the target altitude
while True:
    print("Altitude:", vehicle.location.global_relative_frame.alt)
    if vehicle.location.global_relative_frame.alt >= 4.5:
        print("Target altitude reached")
        break
    time.sleep(1)

for point in waypoints:
    print("Going to:", point)
    vehicle.simple_goto(point)

    # Wait for the drone to reach the target location
    while vehicle.mode.name=="GUIDED":  # Check if still in GUIDED mode
        remaining_distance = vehicle.location.global_frame.distance_to(point)
        print("Distance to target:", remaining_distance)
        if remaining_distance <= 1:
            print("Reached target")
            break

        time.sleep(1)

print("Landing")
vehicle.mode = VehicleMode("LAND")

while vehicle.armed:
    print("Altitude:", vehicle.location.global_relative_frame.alt)
    time.sleep(1)

print("Disarmed")
