
#### build and run docker
```
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive
docker build . -t ardupilot
docker run --rm -it -v `pwd`:/ardupilot ardupilot:latest bash
```
#### inside docker
```
sim_vehicle.py -v copter --console --map -w
```

### drone kit ardupilot

install dronekit
```
pip install dronekit-sitl
pip install mavproxy
```

#### Execution Instructions

Start the Dronekit SITL simulator:
```
dronekit-sitl copter --home=35.98,-9.5.87,0,180
```

#### Run the MAVProxy tool to establish connections:

```
mavproxy.py --master tcp:127.0.0.1:5760 --out udp:127.0.0.1:14551 --out udp:10.55.222.120:14550
```

Execute the Python script to connect to Dronekit:
Note: Ensure Python 3.9 is used for running the script, as later versions might be incompatible with Dronekit.
```
python SuT/ardu/dronekit_mission.py
```

### logs
ArduPilot logs are in .tlog format, you can use mavlogdump.py to convert it to human readable format