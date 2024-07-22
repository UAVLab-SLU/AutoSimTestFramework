# Description

After the drone reache the first waypoint, a batter system failure is injected and the drone is forced to land, the drone did manage to safely land, and a system warning was issued.


```
INFO  [logger] Opened full log file: ./log/2024-07-22/22_30_49.ulg
INFO  [tone_alarm] home set
INFO  [tone_alarm] notify negative
INFO  [commander] Ready for takeoff!
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [commander] Armed by external command
INFO  [tone_alarm] arming warning
INFO  [commander] Takeoff detected
WARN  [battery_simulator] CMD_INJECT_FAILURE, battery empty
WARN  [battery_simulator] CMD_INJECT_FAILURE, battery empty
INFO  [commander] Landing at current position
ERROR [health_and_arming_checks] Preflight Fail: low battery
WARN  [failsafe] Failsafe activated
INFO  [tone_alarm] battery warning (slow)
ERROR [health_and_arming_checks] Preflight Fail: low battery
ERROR [health_and_arming_checks] Preflight Fail: low battery
WARN  [failsafe] Failsafe activated, entering Hold for 5 seconds
INFO  [tone_alarm] battery warning (fast)
WARN  [commander] System does not support shutdown
WARN  [failsafe] Failsafe activated
INFO  [commander] Landing detected
INFO  [commander] Disarmed by external command
INFO  [tone_alarm] notify neutral
INFO  [logger] closed logfile, bytes written: 10692382
INFO  [commander] Connection to ground station lost
ERROR [health_and_arming_checks] Preflight Fail: low battery
```