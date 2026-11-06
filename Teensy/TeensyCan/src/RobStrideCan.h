#pragma once
#include <Arduino.h>
#include <FlexCAN_T4.h>

// Control limits for RobStride-00
constexpr float P_MIN = -12.56637f;
constexpr float P_MAX =  12.56637f;
constexpr float V_MIN = -33.0f;
constexpr float V_MAX =  33.0f;
constexpr float KP_MIN = 0.0f;
constexpr float KP_MAX = 500.0f;
constexpr float KD_MIN = 0.0f;
constexpr float KD_MAX = 5.0f;
constexpr float T_MIN = -14.0f;
constexpr float T_MAX =  14.0f;

struct MotorFeedback {
    float position = 0.0f;
    float velocity = 0.0f;
    float torque   = 0.0f;
    float temp     = 0.0f;
    uint32_t last_update_ms = 0;
};

class RobStrideCan {
public:
    static const uint8_t TOTAL_MOTORS = 1;
    static const uint8_t MASTER_ID = 0xFD;

    float default_kp = 25.0f;
    float default_kd = 0.8f;

    void setup() {
        can1.begin();
        can1.setBaudRate(1000000);
        can1.setMaxMB(16);

        can2.begin();
        can2.setBaudRate(1000000);
        can2.setMaxMB(16);

        delay(50);
        initActuator(1);
    }

    void update() {
        CAN_message_t msg;
        while (can1.read(msg)) {
            parseMessage(msg);
        }
        while (can2.read(msg)) {
            parseMessage(msg);
        }
    }

    void enableAll() {
        for (uint8_t id = 1; id <= TOTAL_MOTORS; id++) {
            setMode(id, 0x03);
        }
    }

    void disableAll() {
        for (uint8_t id = 1; id <= TOTAL_MOTORS; id++) {
            setMode(id, 0x04);
        }
    }

    void setTargetPosition(uint8_t id, float desired_pos, float desired_vel = 0.0f, float ff_torque = 0.0f) {
        if (id < 1 || id > TOTAL_MOTORS) return;

        CAN_message_t msg;
        msg.flags.extended = 1;
        msg.len = 8;

        uint16_t t_int  = pack(ff_torque,   T_MIN,  T_MAX);
        uint16_t p_int  = pack(desired_pos, P_MIN,  P_MAX);
        uint16_t v_int  = pack(desired_vel, V_MIN,  V_MAX);
        uint16_t kp_int = pack(default_kp,  KP_MIN, KP_MAX);
        uint16_t kd_int = pack(default_kd,  KD_MIN, KD_MAX);

        msg.id = (0x01 << 24) | ((uint32_t)t_int << 8) | id;
        msg.buf[0] = p_int >> 8;  msg.buf[1] = p_int & 0xFF;
        msg.buf[2] = v_int >> 8;  msg.buf[3] = v_int & 0xFF;
        msg.buf[4] = kp_int >> 8; msg.buf[5] = kp_int & 0xFF;
        msg.buf[6] = kd_int >> 8; msg.buf[7] = kd_int & 0xFF;

        write(id, msg);
    }

    const MotorFeedback& getFeedback(uint8_t id) const {
        return feedback[constrain(id, 1, TOTAL_MOTORS) - 1];
    }

    bool isHealthy() const {
        uint32_t now = millis();
        for (uint8_t i = 0; i < TOTAL_MOTORS; i++) {
            if (now - feedback[i].last_update_ms > 200) return false;
        }
        return true;
    }

private:
    static FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_32> can1;
    static FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_32> can2;
    static MotorFeedback feedback[TOTAL_MOTORS];

    void initActuator(uint8_t id) {
        // Clear faults
        setMode(id, 0x04);
        delay(20);

        // Set Run Mode to MIT (Param 0x7005 = 0)
        CAN_message_t mode_msg;
        mode_msg.flags.extended = 1;
        mode_msg.id = (0x12 << 24) | ((uint32_t)MASTER_ID << 8) | id;
        mode_msg.len = 8;
        uint8_t param[8] = {0x05, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
        memcpy(mode_msg.buf, param, 8);
        write(id, mode_msg);
        delay(20);

        // Enable
        setMode(id, 0x03);
        delay(20);
    }

    void setMode(uint8_t id, uint8_t type) {
        CAN_message_t msg;
        msg.flags.extended = 1;
        msg.id = (type << 24) | ((uint32_t)MASTER_ID << 8) | id;
        msg.len = 8;
        memset(msg.buf, 0, 8);
        write(id, msg);
    }

    void write(uint8_t id, const CAN_message_t &msg) {
        (id <= 6) ? can1.write(msg) : can2.write(msg);
    }

    static uint16_t pack(float val, float min_v, float max_v) {
        val = constrain(val, min_v, max_v);
        return (uint16_t)((val - min_v) * 65535.0f / (max_v - min_v));
    }

    static float unpack(uint16_t raw, float min_v, float max_v) {
        return ((float)raw / 65535.0f) * (max_v - min_v) + min_v;
    }

    static void parseMessage(const CAN_message_t &msg) {
        uint8_t type = (msg.id >> 24) & 0x1F;
        if (type != 0x02 || msg.len < 8) return;

        // In 0x20001FD, bits 23-8 contain the motor ID (0x0001)
        uint8_t id = (msg.id >> 8) & 0xFF;
        if (id < 1 || id > TOTAL_MOTORS) return;

        auto &fb = feedback[id - 1];
        fb.position = unpack((msg.buf[0] << 8) | msg.buf[1], P_MIN, P_MAX);
        fb.velocity = unpack((msg.buf[2] << 8) | msg.buf[3], V_MIN, V_MAX);
        fb.torque   = unpack((msg.buf[4] << 8) | msg.buf[5], T_MIN, T_MAX);
        fb.temp     = ((int16_t)((msg.buf[6] << 8) | msg.buf[7])) * 0.1f;
        fb.last_update_ms = millis();
    }
};

inline FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_32> RobStrideCan::can1;
inline FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_32> RobStrideCan::can2;
inline MotorFeedback RobStrideCan::feedback[RobStrideCan::TOTAL_MOTORS];