import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)


class PX4PointsMission:
    def __init__(self, points,speed):
        self.points = points
        self.speed = speed
        self.loiter_time = 3


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
            PositionNedYaw(0.0, 0.0, -20.0, 0.0))
        await asyncio.sleep(3)

        for point in self.points:
            print("-- Go to point: " + str(point))
            await drone.offboard.set_position_ned(
                PositionNedYaw(point[0], point[1], point[2], 0))
            await asyncio.sleep(self.loiter_time)

        print("-- Landing")
        await drone.action.land()




if __name__ == "__main__":
    points = [[0.0, 0.0, -10.0], [0.0, 5.0, -10.0]]
    speed = 5
    mission = PX4PointsMission(points, speed)
    mission.start()