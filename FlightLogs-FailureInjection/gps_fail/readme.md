# Description

After reaching each waypoint, a gps signal failure is simulated by injecting a failure command. The drone had not issue for the 1-3 waypoints, but after reaching the 4th waypoint, the drone started to fail. The drone lost control and started drifting. The drone was able to land safely after the failsafe was activated.

For the longer one, gps failure was injected at 4th point the drone entered blind landing mode and drifted a long distance 

```
INFO  [logger] Opened full log file: ./log/2024-07-22/23_10_32.ulg
INFO  [tone_alarm] notify negative
INFO  [commander] Ready for takeoff!
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [commander] Armed by external command
INFO  [tone_alarm] arming warning
INFO  [commander] Takeoff detected
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, GPS off
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, GPS off
WARN  [mc_pos_control] invalid setpoints
WARN  [mc_pos_control] Failsafe: blind land
WARN  [failsafe] Failsafe activated
INFO  [tone_alarm] battery warning (fast)
ERROR [flight_mode_manager] Matching flight task was not able to run, Nav state: 1, Task: 1
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, GPS off
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, GPS off
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, GPS off
WARN  [commander] Switching to AUTO_LAND is currently not available
INFO  [tone_alarm] notify negative
INFO  [tone_alarm] battery warning (fast)
INFO  [commander] Landing detected
INFO  [commander] Disarmed by landing
INFO  [tone_alarm] notify neutral
INFO  [logger] closed logfile, bytes written: 9453255
```


longer one

```
INFO  [logger] [logger] ./log/2024-07-23/17_08_12.ulg
INFO  [mavlink] MAVLink only on localhost (set param MAV_{i}_BROADCAST = 1 to enable network)
INFO  [mavlink] MAVLink only on localhost (set param MAV_{i}_BROADCAST = 1 to enable network)
INFO  [px4] Startup script returned successfully
pxh> INFO  [logger] Opened full log file: ./log/2024-07-23/17_08_12.ulg
INFO  [tone_alarm] home set
INFO  [tone_alarm] notify negative
INFO  [commander] Ready for takeoff!
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [commander] Armed by external command
INFO  [tone_alarm] arming warning
INFO  [commander] Takeoff detected
WARN  [simulator_mavlink] CMD_INJECT_FAILURE, GPS off
WARN  [mc_pos_control] invalid setpoints
WARN  [mc_pos_control] Failsafe: blind land
WARN  [failsafe] Failsafe activated
INFO  [tone_alarm] battery warning (fast)
ERROR [flight_mode_manager] Matching flight task was not able to run, Nav state: 1, Task: 1
WARN  [commander] Switching to AUTO_LAND is currently not available
INFO  [tone_alarm] notify negative
INFO  [tone_alarm] battery warning (fast)
INFO  [commander] Landing detected
INFO  [commander] Disarmed by landing
INFO  [tone_alarm] notify neutral
INFO  [logger] closed logfile, bytes written: 15390155

```