# Description

After reaching each waypoint, a gps signal failure is simulated by injecting a failure command. The drone had not issue for the 1-3 waypoints, but after reaching the 4th waypoint, the drone started to fail. The drone lost control and started drifting. The drone was able to land safely after the failsafe was activated.


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