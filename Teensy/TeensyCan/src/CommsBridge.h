#pragma once
#include <Arduino.h>
#include "RobStrideCan.h"

// Sync words to prevent frame misalignment
constexpr uint32_t CMD_SYNC_HEADER   = 0xAA55AA55;
constexpr uint32_t TELEM_SYNC_HEADER = 0x55AA55AA;

// Host (PC/Jetson) -> Teensy (50 Hz action policy)
struct __attribute__((packed)) PolicyCommand {
  uint32_t header = CMD_SYNC_HEADER;
  float target_positions[RobStrideCan::TOTAL_MOTORS];   // Desired joint angles (rad)
};

// Teensy -> Host (PC/Jetson) (50 Hz observation buffer)
struct __attribute__((packed)) ObservationTelemetry {
  uint32_t header = TELEM_SYNC_HEADER;
  float positions[RobStrideCan::TOTAL_MOTORS];          // Measured joint angles (rad)
  float velocities[RobStrideCan::TOTAL_MOTORS];         // Measured joint speeds (rad/s)
  float torques[RobStrideCan::TOTAL_MOTORS];            // Measured joint torques (Nm)
  uint8_t is_healthy;                                   // 1 = all motors responsive, 0 = timeout
};


class CommsBridge {
public:
  PolicyCommand cmd;
  ObservationTelemetry telem;


  void setup(uint32_t baud_rate = 1000000) {
    Serial.begin(baud_rate);
    while (!Serial && millis() < 2000);
  }


  // Polls USB serial stream, aligns sync bytes, and unpacks command
  bool receiveCommand() {
    bool updated = false;

    while (Serial.available() >= (int)sizeof(PolicyCommand)) {
      // In little-endian, uint32_t 0xAA55AA55 starts with byte 0x55
      if (Serial.peek() != 0x55) {
        Serial.read(); // Discard noise until frame start
        continue;
      }

      PolicyCommand temp;
      Serial.readBytes((char*)&temp, sizeof(PolicyCommand));

      if (temp.header == CMD_SYNC_HEADER) {
        memcpy(&cmd, &temp, sizeof(PolicyCommand));
        updated = true;

      // Blink onboard LED when command received
      digitalWriteFast(LED_BUILTIN, !digitalReadFast(LED_BUILTIN));
      }
    }

    return updated;
  }


  // Packs motor feedback and transmits telemetry frame over USB
  void sendTelemetry(const RobStrideCan &robot) {
    telem.is_healthy = robot.isHealthy() ? 1 : 0;

    for (uint8_t i = 0; i < RobStrideCan::TOTAL_MOTORS; i++) {
      const auto &fb = robot.getFeedback(i + 1);
      telem.positions[i]   = fb.position;
      telem.velocities[i]  = fb.velocity;
      telem.torques[i] = fb.torque;
    }

    Serial.write((uint8_t*)&telem, sizeof(ObservationTelemetry));
  }
};