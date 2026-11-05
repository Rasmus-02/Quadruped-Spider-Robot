import struct
import time
import numpy as np
import serial

TOTAL_MOTORS = 1

# Formats matching Teensy structs
CMD_FMT = f"<I{TOTAL_MOTORS}f"       # uint32 header + N float32 target positions
TELEM_FMT = f"<I{TOTAL_MOTORS * 3}fB"  # uint32 header + 3N float32 (pos, vel, tau) + uint8 is_healthy

CMD_HEADER = 0xAA55AA55
TELEM_HEADER = 0x55AA55AA
TELEM_SYNC_BYTES = struct.pack("<I", TELEM_HEADER)  # b'\xaa\x55\xaa\x55'

CMD_SIZE = struct.calcsize(CMD_FMT)
TELEM_SIZE = struct.calcsize(TELEM_FMT)


class MotorController:

    def __init__(self, port="/dev/ttyACM0", baudrate=1000000):
        self.ser = serial.Serial(port, baudrate, timeout=0.01, write_timeout=0.01)
        time.sleep(0.5)
        self.ser.reset_input_buffer()  # Flush stale bytes received during boot

        self.positions = np.zeros(TOTAL_MOTORS, dtype=np.float32)
        self.velocities = np.zeros(TOTAL_MOTORS, dtype=np.float32)
        self.torques = np.zeros(TOTAL_MOTORS, dtype=np.float32)
        self.is_healthy = False
        self._rx_buf = bytearray()


    def send_action(self, target_positions):
        """Sends target joint angles (rad) to MotorController (Teensy).
            Args:
            target_positions (iterable of float): Array containing target joint positions in radians, ordered according to motor id.
        """
        if len(target_positions) != TOTAL_MOTORS:
            raise ValueError(f"Expected {TOTAL_MOTORS} target positions, got {len(target_positions)}")
            
        payload = struct.pack(CMD_FMT, CMD_HEADER, *target_positions)
        try:
            self.ser.write(payload)
        except serial.SerialTimeoutException:
            pass


    def get_observation(self):
        """Reads latest telemetry without blocking."""
        # Read all currently available bytes in one non-blocking call
        waiting = self.ser.in_waiting
        if waiting > 0:
            self._rx_buf.extend(self.ser.read(waiting))

        # Find the most recent complete packet in the buffer
        while len(self._rx_buf) >= TELEM_SIZE:
            idx = self._rx_buf.find(TELEM_SYNC_BYTES)

            if idx == -1:
                # Header not found, keep last 3 bytes in case header is split
                self._rx_buf = self._rx_buf[-3:]
                break

            if idx > 0:
                # Discard noise bytes before header
                del self._rx_buf[:idx]

            if len(self._rx_buf) < TELEM_SIZE:
                # Wait for the rest of the frame to arrive
                break

            # Extract and unpack valid frame
            frame = self._rx_buf[:TELEM_SIZE]
            del self._rx_buf[:TELEM_SIZE]

            unpacked = struct.unpack(TELEM_FMT, frame)
            
            pos_end = 1 + TOTAL_MOTORS
            vel_end = pos_end + TOTAL_MOTORS
            tau_end = vel_end + TOTAL_MOTORS
            
            self.positions[:] = unpacked[1:pos_end]
            self.velocities[:] = unpacked[pos_end:vel_end]
            self.torques[:] = unpacked[vel_end:tau_end]
            self.is_healthy = bool(unpacked[tau_end])

        return self.positions, self.velocities, self.torques, self.is_healthy

    
    def close(self):
        self.ser.close()