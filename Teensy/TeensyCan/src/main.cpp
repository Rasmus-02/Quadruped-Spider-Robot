#include <Arduino.h>
#include "RobStrideCan.h"
#include "CommsBridge.h"

#define MOCK_MODE 0

RobStrideCan motor;
CommsBridge  comms;

uint32_t last_can_send_us  = 0;
uint32_t last_host_sync_ms = 0;

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    comms.setup(1000000);

#if !MOCK_MODE
    motor.default_kp = 25.0f;
    motor.default_kd = 0.8f;
    motor.setup();

    uint32_t start_wait = millis();
    while (millis() - start_wait < 300) {
        motor.update();
        delay(2);
    }
#endif
}

void loop() {
#if !MOCK_MODE
    motor.update();
    comms.receiveCommand();

    // 500 Hz control loop
    uint32_t current_us = micros();
    if (current_us - last_can_send_us >= 2000) {
        last_can_send_us = current_us;

        if (motor.isHealthy() || millis() < 2000) {
            for (uint8_t id = 1; id <= RobStrideCan::TOTAL_MOTORS; id++) {
                motor.setTargetPosition(id, comms.cmd.target_positions[id - 1]);
            }
        } else {
            static uint32_t last_retry = 0;
            if (current_us - last_retry > 500000) {
                motor.enableAll();
                last_retry = current_us;
            }
        }
    }

    // 100 Hz host telemetry
    uint32_t current_ms = millis();
    if (current_ms - last_host_sync_ms >= 10) {
        last_host_sync_ms = current_ms;
        comms.sendTelemetry(motor);
    }

#else
    comms.receiveCommand();
    uint32_t current_ms = millis();
    if (current_ms - last_host_sync_ms >= 10) {
        last_host_sync_ms = current_ms;
        comms.telem.is_healthy = 1;
        for (uint8_t i = 0; i < RobStrideCan::TOTAL_MOTORS; i++) {
            comms.telem.positions[i]  = comms.cmd.target_positions[i];
            comms.telem.velocities[i] = 0.0f;
            comms.telem.torques[i]    = 0.0f;
        }
        Serial.write((uint8_t*)&comms.telem, sizeof(ObservationTelemetry));
    }
#endif
}