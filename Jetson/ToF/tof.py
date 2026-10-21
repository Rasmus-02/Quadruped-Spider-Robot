import time
import board
import busio
import adafruit_vl53l1x

i2c = busio.I2C(board.SCL, board.SDA)

# Initialize sensor
sensor = adafruit_vl53l1x.VL53L1X(i2c)
sensor.distance_mode = 1
sensor.start_ranging()

# Scan loop
try:
    while True:
        if sensor.data_ready:
            distance = sensor.distance
            if distance is not None:
                print(distance)
            sensor.clear_interrupt()

except KeyboardInterrupt:
    sensor.stop_ranging