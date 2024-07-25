import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.failure import FailureUnit, FailureType
from mavsdk.telemetry import LandedState

"""
{
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "ClockType": "SteppableClock",
    "ViewMode": "SpringArmChase",
    "DefaultVehicleConfig": "PX4",
    "Vehicles": {
        "Drone 1": {
            "X": 0.0,
            "Y": 0.0,
            "Z": -3,
            "LocalHostIp": "192.168.1.181",
            "VehicleType": "PX4Multirotor",
            "UseSerial": false,
            "LockStep": true,
            "UseTcp": true,
            "TcpPort": 4560,
            "ControlPortLocal": 14540,
            "ControlPortRemote": 14580,
            "Sensors": {
                "Barometer": {
                    "SensorType": 1,
                    "Enabled": true,
                    "PressureFactorSigma": 0.0001825
                }
            },
            "Parameters": {
                "NAV_RCL_ACT": 0,
                "NAV_DLL_ACT": 0,
                "COM_OBL_ACT": 1,
                "LPE_LAT": 40.7128,
                "LPE_LON": -74.0060
            }
        }
    },
    "Wind": { "X": 10.6, "Y": 10.6, "Z": 0 }
}
"""

class PX4PointsMission:
    def __init__(self, points, speed):
        self.points = points
        self.speed = speed
        self.loiter_time = 10


    def __str__(self):
        return ("Points: " + str(self.points) +
                "\n" + "Speed: " + str(self.speed) +
                "\n" + "Loiter Time: " + str(self.loiter_time))

    def start(self):
        print("Starting mission with the following parameters:")
        print(self)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(self.run())
        # waiting for the mission to finish
        loop.run_until_complete(task)

    async def run(self):
        print("Connecting to mavsdk server")
        drone = System()
        print("Connecting to drone")
        await drone.connect(system_address="udp://127.0.0.1:14580")

        print("Waiting for drone to connect...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                break

        print("Waiting for drone to have a global position estimate...")
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- Global position estimate OK")
                break

        print("-- Arming")
        while True:
            try:
                await drone.action.arm()
                break
            except OffboardError as error:
                print(f"Arming failed with error code: {error._result.result}")
                await asyncio.sleep(1)

        print("-- Setting initial setpoint")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

        print("-- Starting offboard")
        try:
            await drone.offboard.start()
            print("-- Offboard started")
        except OffboardError as error:
            print(f"Starting offboard mode failed \
                            with error code: {error._result.result}")
            print("-- Disarming")
            await drone.action.disarm()
            return

        await asyncio.sleep(1)
        print("-- test telemetry")
        print(drone.telemetry.health())
        await asyncio.sleep(1)

        await drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 0.0, -5.0, 0.0))
        await asyncio.sleep(3)


        for point in self.points:
            print("-- Go to point: " + str(point))
            await drone.offboard.set_position_ned(
                PositionNedYaw(point[0], point[1], point[2], 0))
            await asyncio.sleep(self.loiter_time)

        print("-- Landing")
        await drone.action.land()

        # wait for the drone to land
        async for landed_state in drone.telemetry.landed_state():
            if landed_state == LandedState.IN_AIR:
                await asyncio.sleep(1)
            elif landed_state == LandedState.ON_GROUND:
                print("-- Landed")
                break

        try:
            await drone.action.disarm()
            print("-- Disarmed")
        except OffboardError as error:
            print(f"Disarming failed with error code: {error._result.result}")

if __name__ == "__main__":
    points =  [
        [0, 0, 0],
        [0, 10, -5],
        [10, 10, -5],
        [10, 0, -5],
        [0, 0, -5]
      ]
    speed = 45
    mission = PX4PointsMission(points, speed)
    mission.start()
