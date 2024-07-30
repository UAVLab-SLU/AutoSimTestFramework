# Description

5 waypoints, after reach 4th, a AirSpeed Off failure was injected, the drone proceeded to the last waypoint and landed without issue. no visual issues were observed.

```
INFO  [logger] [logger] ./log/2024-07-30/20_42_47.ulg
INFO  [logger] Opened full log file: ./log/2024-07-30/20_42_47.ulg
INFO  [mavlink] MAVLink only on localhost (set param MAV_{i}_BROADCAST = 1 to enable network)
INFO  [mavlink] MAVLink only on localhost (set param MAV_{i}_BROADCAST = 1 to enable network)
INFO  [px4] Startup script returned successfully
pxh> INFO  [tone_alarm] home set
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [tone_alarm] notify negative
INFO  [commander] Ready for takeoff!
INFO  [tone_alarm] notify positive
INFO  [commander] Armed by external command
INFO  [navigator] Executing Mission
INFO  [tone_alarm] arming warning
INFO  [navigator] Climb to 50.0 meters above home
INFO  [commander] Takeoff detected
INFO  [mavlink] partner IP: 127.0.0.1
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, airspeed off
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, airspeed off
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, airspeed off
INFO  [commander] Returning to launch
INFO  [navigator] RTL: return at 172 m (50 m above destination)
INFO  [navigator] RTL: descend to 133 m (10 m above destination)
INFO  [navigator] RTL: land at destination
INFO  [commander] Landing detected
INFO  [commander] Disarmed by landing
INFO  [tone_alarm] notify neutral
INFO  [logger] closed logfile, bytes written: 34766454
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, airspeed off

PX4 Exiting...
pxh> Exiting NOW.
ninja: build stopped: interrupted by user.
make: *** [Makefile:227: px4_sitl_default] Interrupt
```