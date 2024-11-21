import asyncio
from mavsdk import System
from mavsdk.failure import FailureUnit, FailureType

drone = System()
print("Connecting to drone")
async def connect():
    await drone.connect(system_address="udp://127.0.0.1:14580")
    await drone.failure.inject(
                    FailureUnit.SENSOR_AIRSPEED,
                    FailureType.OFF,
                    0
                    )

asyncio.run(connect())

