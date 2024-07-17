# AutoSimTestFramework
LLM Agents driven automation of sUAS simulation testing


## Flight Control relay
Description:
`llm_relay` contains the code for the PX4 and ardu relay. The relay is responsible for communicating with the flight controller firmware using Mavsdk server.


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
```
python llm_relay/ardu/dronekit_mission.py
```