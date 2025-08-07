import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    pipeline.start(config)
    print("Pipeline started.")
    frames = pipeline.wait_for_frames()
    print("Received frames.")
    pipeline.stop()
except Exception as e:
    print("Error:", e)