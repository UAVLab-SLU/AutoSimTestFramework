# Description

5 waypoints, after reach 4th, a Barometer Off failure was injected, the drone proceeded to the last waypoint and failed to land, a manual landing command was issued.


```
INFO  [logger] Opened full log file: ./log/2024-07-30/15_43_28.ulg
INFO  [tone_alarm] home set
INFO  [tone_alarm] notify positive
INFO  [commander] Ready for takeoff!
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [tone_alarm] notify negative
INFO  [tone_alarm] notify negative
INFO  [tone_alarm] notify negative
INFO  [tone_alarm] notify negative
INFO  [tone_alarm] notify negative
INFO  [tone_alarm] notify positive
INFO  [commander] Armed by external command
INFO  [tone_alarm] arming warning
INFO  [navigator] Executing Mission
INFO  [navigator] Climb to 50.0 meters above home
INFO  [commander] Takeoff detected
INFO  [mavlink] partner IP: 127.0.0.1
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, baro off
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
WARN  [health_and_arming_checks] Preflight Fail: No valid data from Baro 0
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
INFO  [navigator] Mission finished, loitering
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
INFO  [commander] Landing detected
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
INFO  [commander] Disarmed by landing
INFO  [tone_alarm] notify neutral
INFO  [logger] closed logfile, bytes written: 63541662
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!
ERROR [vehicle_air_data] BARO #0 failed:  TIMEOUT!

```