import time
import math
import sys
import termios
import numpy as np
from motor_controller import MotorController
from pynput import keyboard

JOINT_SPEED = 0.005
# Degrees of motion (+- range/2)
HIP_Z_RANGE = math.radians(60)
HIP_X_RANGE = math.radians(40)
KNEE_X_RANGE = math.radians(30)

SEPERATOR = "\033[96m|\033[0m" # Cyan colored seperators in terminal output

pressed_keys: set[str] = set()


# Key pressed
def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    if isinstance(key, keyboard.KeyCode) and key.char:
        pressed_keys.add(key.char.lower())


# Key released
def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    if isinstance(key, keyboard.KeyCode) and key.char:
        pressed_keys.discard(key.char.lower())


# Handle keyboard input
def get_commands() -> tuple[float, float, float]:
    hip_z = float('a' in pressed_keys) - float('d' in pressed_keys)
    hip_x = float('w' in pressed_keys) - float('s' in pressed_keys)
    knee_x = float('q' in pressed_keys) - float('e' in pressed_keys)
    return hip_z * JOINT_SPEED, hip_x * JOINT_SPEED, knee_x * JOINT_SPEED


def test_pipeline():
    port = "/dev/ttyACM0"
    print(f"Connecting to Teensy on {port}...")
    controller = MotorController(port=port)
    print("Connected! Streaming simulated actions to Teensy...\n")

    # Timer & Keyboard listener
    t_start = time.time()
    last_print = time.time()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Disable terminal input prints
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    new_term = termios.tcgetattr(fd)
    new_term[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, new_term)

    try:
        target_hip_z: float = 0.0
        target_hip_x: float = 0.0
        target_knee_x: float = 0.0

        while True:
            t = time.time() - t_start

            # Generate Can movement command for motors (+ padding for unconnected motors)
            h_z, h_x, k_x = get_commands()

            target_hip_z = max(-HIP_Z_RANGE / 2.0, min(target_hip_z + h_z, HIP_Z_RANGE / 2.0))
            target_hip_x = max(-HIP_X_RANGE / 2.0, min(target_hip_x + h_x, HIP_X_RANGE / 2.0))
            target_knee_x = max(-KNEE_X_RANGE / 2.0, min(target_knee_x + k_x, KNEE_X_RANGE / 2.0))

            test_positions = np.array([
                target_hip_z,
                target_hip_x,
                target_knee_x,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # Unused motors
            ], dtype=np.float32)

            # Send action to Teensy
            controller.send_action(test_positions)

            # Read echoed observation
            pos, vel, tau, healthy = controller.get_observation()

            # Print 5 times per second
            if time.time() - last_print >= 0.2:
                last_print = time.time()
                print(f"[Healthy: {healthy}] "
                      f"{SEPERATOR} CMD: {test_positions[0]:+6.2f}, {test_positions[1]:+6.2f}, {test_positions[2]:+6.2f} {SEPERATOR} "
                      f"HIP X POS: {pos[0]:+6.2f} rad {SEPERATOR} "
                      f"HIP Z POS: {pos[1]:+6.2f} rad {SEPERATOR} "
                      f"Knee X POS: {pos[2]:+6.2f} rad {SEPERATOR} "
                      f"Buffer waiting: {controller.ser.in_waiting} bytes")

            time.sleep(0.02) # 50 Hz loop

    except KeyboardInterrupt:
        print("\nStopping.")
        controller.close()
        listener.stop()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        


if __name__ == "__main__":
    test_pipeline()