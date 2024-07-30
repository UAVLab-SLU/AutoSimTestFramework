# Description

5 waypoints, after reach 4th, a Magnetometer Off failure was injected, the drone proceeded to the last waypoint and landed without issue. no visual issues were observed.

```
INFO  [logger] Opened full log file: ./log/2024-07-30/15_50_31.ulg
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [tone_alarm] home set
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [tone_alarm] notify positive
INFO  [tone_alarm] notify negative
INFO  [commander] Ready for takeoff!
INFO  [tone_alarm] notify positive
INFO  [commander] Armed by external command
INFO  [navigator] Executing Mission
INFO  [tone_alarm] arming warning
INFO  [navigator] Climb to 50.0 meters above home
INFO  [commander] Takeoff detected
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, mag 0 off
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, mag 1 off
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
WARN  [health_and_arming_checks] Preflight Fail: No valid data from Compass 0
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
ERROR [vehicle_magnetometer] MAG #0 failed:  TIMEOUT!
INFO  [commander] Landing detected
INFO  [navigator] Mission finished, landed
INFO  [commander] Disarmed by landing
INFO  [tone_alarm] notify neutral
INFO  [logger] closed logfile, bytes written: 42480107

```