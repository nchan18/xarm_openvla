import pyrealsense2 as rs

# Create a context object to manage devices
context = rs.context()

# List all connected devices
devices = context.query_devices()

# Iterate through devices and print their serial numbers
for device in devices:
    print("Device Serial Number:", device.get_info(rs.camera_info.serial_number))