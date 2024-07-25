# Description

Mavlink signal disabled mid air, drone detected the signal loss and activated failsafe mode. The drone landed safely.


```
INFO  [logger] Opened full log file: ./log/2024-07-22/23_19_29.ulg
INFO  [mavlink] MAVLink only on localhost (set param MAV_{i}_BROADCAST = 1 to enable network)
INFO  [mavlink] MAVLink only on localhost (set param MAV_{i}_BROADCAST = 1 to enable network)
INFO  [px4] Startup script returned successfully
pxh> INFO  [tone_alarm] notify negative
INFO  [commander] Ready for takeoff!
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [commander] Armed by external command
INFO  [tone_alarm] arming warning
INFO  [commander] Takeoff detected
WARN  [failsafe] Failsafe activated
INFO  [tone_alarm] battery warning (fast)
ERROR [flight_mode_manager] Matching flight task was not able to run, Nav state: 2, Task: 1
INFO  [commander] Landing detected
INFO  [commander] Disarmed by landing
INFO  [tone_alarm] notify neutral
INFO  [logger] closed logfile, bytes written: 41091060

```