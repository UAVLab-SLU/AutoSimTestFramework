import socket

import dronekit
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time


def discover_connection():
    # Common Ports Used by ArduPilot SITL
    ports = [5501, 14550, 14551, 5760, 5762, 5763]  # Add more if needed
    ips = ["127.0.0.1", "localhost"]

    for ip in ips:
        for port in ports:
            print(f"Trying connection on {ip}:{port}")
            try:
                vehicle = connect(f'udp:{ip}:{port}', wait_ready=False, timeout=3)
                vehicle.close()
                return f'udp:{ip}:{port}'
            except Exception as e:
                pass

            try:
                vehicle = connect(f'tcp:{ip}:{port}', wait_ready=False, timeout=3)
                vehicle.close()
                return f'tcp:{ip}:{port}'
            except Exception as e:
                pass
    # If no connection is found
    raise Exception("No valid connection found.")


# Discover the connection
connection_string = discover_connection()
print(f"Found connection: {connection_string}")

# Connect to the vehicle using the discovered connection string
vehicle = connect(connection_string, wait_ready=True)
print("Connected to the vehicle!")

# ... (rest of your flight code remains the same)
