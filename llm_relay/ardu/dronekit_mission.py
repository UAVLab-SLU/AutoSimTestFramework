from dronekit import connect, VehicleMode
import time

# Connect to the Vehicle
vehicle = connect('udp:127.0.0.1:14551',wait_ready=True)

waypoints = [
        [0, 0, 0],
        [0, 10, -5],
        [10, 10, -5],
        [10, 0, -5],
        [0, 0, -5]
      ]

latitude = -35.363261
longitude = 149.165230

# set location
vehicle.home_location = vehicle.location.global_frame
vehicle.home_location.lat = latitude
vehicle.home_location.lon = longitude


print("arming the vehicle")
while not vehicle.is_armable:
    print("waiting for vehicle to become armable")
    time.sleep(1)

vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True

print("takeoff")

vehicle.simple_takeoff(10)

while True:
    print("Altitude: ", vehicle.location.global_relative_frame.alt)
    if vehicle.location.global_relative_frame.alt >= 9.5:
        print("target altitude reached")
        break
    time.sleep(1)


for point in waypoints:
    print("going to point: ", point)
    vehicle.simple_goto(point)
    while True:
        print("current location: ", vehicle.location.global_relative_frame)
        print("target location: ", point)
        distance = vehicle.location.global_relative_frame.distance_to(point)
        print("distance to target: ", distance)
        if distance <= 1:
            print("target reached")
            break
        time.sleep(1)



print("landing")
vehicle.mode = VehicleMode("LAND")

while vehicle.location.global_relative_frame.alt >= 0.5:
    print("Altitude: ", vehicle.location.global_relative_frame.alt)
    time.sleep(1)

print("disarming")
vehicle.armed = False


