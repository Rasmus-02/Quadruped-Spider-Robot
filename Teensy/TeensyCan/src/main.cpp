#include <Arduino.h>
#include "RobStrideCan.h"
#include "CommsBridge.h"

// Used for testing without motors
#define MOCK_MODE 1

RobStrideCan motor;
CommsBridge  comms;

uint32_t last_can_send_us  = 0;
uint32_t last_host_sync_ms = 0;


void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    comms.setup(1000000);

#if !MOCK_MODE
    // TODO: Match default actuator PD gains with Isaac Lab config
    motor.default_kp = 25.0f;
    motor.default_kd = 0.8f;

    // Setup & enable motors
    motor.setup();
    delay(100);
    motor.enableAll();

    // Wait up to 500ms and listen to CAN events to get initial motor feedback frames
    uint32_t start_wait = millis();
    while (millis() - start_wait < 500) {
        motor.update();
        delay(5);
    }
#endif
}


void loop() {
// Real Hardware
#if !MOCK_MODE
    // Feedback: Motors -> Teensy
    motor.update();

    // CMD: Jetson/PC -> Teensy
    comms.receiveCommand();

    // Motor control loop (500 Hz / every 2000 us)
    uint32_t current_us = micros();
    if (current_us - last_can_send_us >= 2000) {
        last_can_send_us = current_us;

        if (motor.isHealthy()) {
            for (uint8_t id = 1; id <= RobStrideCan::TOTAL_MOTORS; id++) {
                motor.setTargetPosition(id, comms.cmd.target_positions[id - 1]);
            }
        } 
        else {
            motor.disableAll();
        }
    }

    // Stream observation telemetry back to host (100 Hz / every 10 ms)
    uint32_t current_ms = millis();
    if (current_ms - last_host_sync_ms >= 10) {
        last_host_sync_ms = current_ms;
        comms.sendTelemetry(motor);
    }

// Test simulation
#else
    comms.receiveCommand();

    uint32_t current_ms = millis();
    if (current_ms - last_host_sync_ms >= 20) { // 50 Hz telemetry stream
        last_host_sync_ms = current_ms;

        // Populate fake telemetry reflecting received commands
        comms.telem.is_healthy = 1;
        for (uint8_t i = 0; i < RobStrideCan::TOTAL_MOTORS; i++) {
            // Simulate motor moving toward target position
            comms.telem.positions[i]  = comms.cmd.target_positions[i];
            comms.telem.velocities[i] = 0.12f * (i + 1); // Mock speed
            comms.telem.torques[i]    = 0.5f;             // Mock torque
        }

        Serial.write((uint8_t*)&comms.telem, sizeof(ObservationTelemetry));
    }
#endif
}