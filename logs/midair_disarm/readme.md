# Description

After reaching each waypoint, a simulated motor failure is injected into the system. The system detected the failure but continue to the next waypoint, I think the mavlink safety switch disallows mid-air disarm.


```
INFO  [logger] Opened full log file: ./log/2024-07-22/22_35_42.ulg
INFO  [mavlink] MAVLink only on localhost (set param MAV_{i}_BROADCAST = 1 to enable network)
INFO  [mavlink] MAVLink only on localhost (set param MAV_{i}_BROADCAST = 1 to enable network)
INFO  [px4] Startup script returned successfully
pxh> INFO  [tone_alarm] notify negative
INFO  [commander] Ready for takeoff!
INFO  [mavlink] partner IP: 127.0.0.1
INFO  [commander] Armed by external command
INFO  [tone_alarm] arming warning
INFO  [commander] Takeoff detected
WARN  [failure_detector] CMD_INJECT_FAILURE, motors off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 0 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 1 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 2 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 3 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 4 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 5 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 6 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 7 off
WARN  [health_and_arming_checks] Preflight Fail: Motor failure detected
WARN  [failsafe] Failsafe activated
INFO  [tone_alarm] battery warning (fast)
WARN  [failure_detector] CMD_INJECT_FAILURE, motors off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 0 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 1 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 2 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 3 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 4 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 5 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 6 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 7 off
WARN  [failure_detector] CMD_INJECT_FAILURE, motors off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 0 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 1 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 2 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 3 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 4 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 5 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 6 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 7 off
WARN  [failure_detector] CMD_INJECT_FAILURE, motors off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 0 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 1 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 2 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 3 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 4 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 5 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 6 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 7 off
WARN  [failure_detector] CMD_INJECT_FAILURE, motors off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 0 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 1 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 2 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 3 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 4 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 5 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 6 off
INFO  [failure_detector] CMD_INJECT_FAILURE, motor 7 off
INFO  [commander] Landing at current position
WARN  [health_and_arming_checks] Preflight Fail: Motor failure detected
INFO  [commander] Landing detected
INFO  [commander] Disarmed by external command
INFO  [tone_alarm] notify neutral
INFO  [logger] closed logfile, bytes written: 12252048
INFO  [commander] Connection to ground station lost

```