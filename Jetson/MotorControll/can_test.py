import time
import numpy as np
from MotorController import MotorController


def test_pipeline():
    port = "/dev/ttyACM0"
    print(f"Connecting to Teensy on {port}...")
    controller = MotorController(port=port)
    print("Connected! Streaming simulated actions to Teensy...\n")

    t_start = time.time()
    last_print = time.time()

    try:
        while True:
            t = time.time() - t_start

            # Generate sine wave targets for all 12 joints
            test_positions = np.array([
                np.sin(t * 2.0 + i * 0.5) for i in range(12)
            ], dtype=np.float32)

            # Send action to Teensy
            controller.send_action(test_positions)

            # Read echoed observation
            pos, vel, tau, healthy = controller.get_observation()

            # 3. Print cleanly 5 times per second (using standard newlines)
            if time.time() - last_print >= 0.2:
                last_print = time.time()
                print(f"[Healthy: {healthy}] "
                      f"CMD: {test_positions[0]:+6.2f} rad | "
                      f"MOTOR POS: {pos[0]:+6.2f} rad | "
                      f"MOTOR VEL: {vel[0]:+5.2f} rad/s | "
                      f"Buffer waiting: {controller.ser.in_waiting} bytes")

            time.sleep(0.02) # 50 Hz loop

    except KeyboardInterrupt:
        print("\nStopping.")
        controller.close()

if __name__ == "__main__":
    test_pipeline()