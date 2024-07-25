from dronekit import connect, VehicleMode,  LocationGlobalRelative, LocationGlobal
import time, math
def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two `LocationGlobal` or `LocationGlobalRelative` objects.
    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5
# Connect to the Vehicle
vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)
print("Connected to the vehicle!")

# get current geo-coordinates
print("Current position: ", vehicle.location.global_frame)

lat = vehicle.location.global_frame.lat
lon = vehicle.location.global_frame.lon
alt = vehicle.location.global_frame.alt


waypoints = [
    LocationGlobalRelative(lat + 0.001, lon, 5),
    LocationGlobalRelative(lat + 0.001, lon + 0.001, 5),
    LocationGlobalRelative(lat, lon + 0.001, 5),
    LocationGlobalRelative(lat, lon, 5)
]

wind_vector = [3.5, 3.5, 0]
vehicle.airspeed = 3


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
    # Wait for the drone to get close to the target point
    while True:
        dist = get_distance_metres(vehicle.location.global_frame, point)
        print("Distance to target:", dist)
        if dist <= 2:
            print("Reached target")
            break
        time.sleep(1)


print("Landing")
vehicle.mode = VehicleMode("LAND")

while vehicle.armed:
    print("Altitude:", vehicle.location.global_relative_frame.alt)
    time.sleep(1)

print("Disarmed")
