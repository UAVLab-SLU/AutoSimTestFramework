
### Execution Instructions
```bash
git clone https://github.com/PX4/PX4-Autopilot.git Firmware
cd Firmware
git submodule update --init --recursive
```


replace the IP address with the IP address of the host machine
```bash
cd Firmware
export PX4_SIM_HOST_ADDR=192.168.1.181
make px4_sitl_default none_iris
```