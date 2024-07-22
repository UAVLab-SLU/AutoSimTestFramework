# AutoSimTestFramework
LLM Agents driven automation of sUAS simulation testing


## Flight Control relay
Description:
`llm_relay` contains the code for the PX4 and ardu relay. The relay is responsible for communicating with the flight controller firmware using Mavsdk server.

## Input Mission format

Waypoint based mission
first param is the speed of the drone
second param is the list of waypoints
```json
{
            "Mission": {
                "name": "Search_and_Rescue_Mission",
                "param": [
                    20,
                    [
                        [
                            0,
                            0,
                            0
                        ],
                        [
                            0,
                            10,
                            -5
                        ],
                        [
                            10,
                            10,
                            -5
                        ],
                        [
                            10,
                            0,
                            -5
                        ],
                        [
                            0,
                            10,
                            -5
                        ],
                        [
                            0,
                            0,
                            0
                        ]
                    ]
                ]
            }
        }
```

Automated misson
```
{
    "Mission": {
        "name": "Search_and_Rescue_Mission",
        "mode": "auto",
        "center": [0, 0, 0],
        "radius": 10,
        "height": 5
    }
}
```
## PX4
replace the IP address with the IP address of the host machine
```bash
cd Firmware
export PX4_SIM_HOST_ADDR=192.168.1.181
make px4_sitl_default none_iris
```

## ardupilot

### build and run docker
```
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive
docker build . -t ardupilot
docker run --rm -it -v `pwd`:/ardupilot ardupilot:latest bash
```
### inside docker
```
sim_vehicle.py -v copter --console --map -w
```


## drone kit ardupilot

install dronekit
```
pip install dronekit-sitl
pip install mavproxy
```

start 
```
dronekit-sitl copter --home=35.98,-9.5.87,0,180
```
```
mavproxy.py --master tcp:127.0.0.1:5760 --out udp:127.0.0.1:14551 --out udp:10.55.222.120:14550
```

then use python script to connect to the dronekit
Note: this script need python 3.9 to run, 3.10+ will not work due to syntax in dronekit
```
python llm_relay/ardu/dronekit_mission.py
```

## logs
ArduPilot logs are in .tlog format, you can use mavlogdump.py to convert it to human readable format
```