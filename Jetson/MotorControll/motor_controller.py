import struct
import time
import numpy as np
import serial

# Formats matching Teensy structs
CMD_FMT = "<I12f"     # uint32 header + 12 float32 target positions
TELEM_FMT = "<I36fB"  # uint32 header + 36 float32 (pos, vel, tau) + uint8 is_healthy

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

        self.positions = np.zeros(12, dtype=np.float32)
        self.velocities = np.zeros(12, dtype=np.float32)
        self.torques = np.zeros(12, dtype=np.float32)
        self.is_healthy = False
        self._rx_buf = bytearray()


    def send_action(self, target_positions):
        """Sends target joint angles (rad) to MotorController (Teensy).
            Args:
            target_positions (iterable of float): Array containing 12 target joint target positions in radians, ordered according to motor id.
        """
        if len(target_positions) != 12:
            raise ValueError(f"Expected 12 target positions, got {len(target_positions)}")
            
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
                # Header not found; keep last 3 bytes in case header is split
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
            self.positions[:] = unpacked[1:13]
            self.velocities[:] = unpacked[13:25]
            self.torques[:] = unpacked[25:37]
            self.is_healthy = bool(unpacked[37])

        return self.positions, self.velocities, self.torques, self.is_healthy

    
    def close(self):
        self.ser.close()