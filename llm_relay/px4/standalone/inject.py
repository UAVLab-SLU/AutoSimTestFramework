import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.failure import FailureUnit, FailureType
from mavsdk.telemetry import LandedState

drone = System()
print("Connecting to drone")
async def connect():
    await drone.connect(system_address="udp://127.0.0.1:14580")
    await drone.failure.inject(
                    FailureUnit.SENSOR_AIRSPEED,  # The unit to fail
                    FailureType.OFF,
                    0
                    )

asyncio.run(connect())

