#pragma once
#include <Arduino.h>
#include <FlexCAN_T4.h>

// Control limits for RobStride-00 (safety)
#define P_MIN  -12.56637f
#define P_MAX   12.56637f
#define V_MIN  -33.0f
#define V_MAX   33.0f
#define KP_MIN  0.0f
#define KP_MAX  500.0f
#define KD_MIN  0.0f
#define KD_MAX  5.0f
#define T_MIN  -14.0f
#define T_MAX   14.0f

// Motor -> Teensy (Used for RL observation)
struct MotorFeedback {
  float position  = 0.0f; // rad
  float velocity  = 0.0f; // rad/s
  float torque    = 0.0f; // Nm
  float temp      = 0.0f; // Celsius
  uint32_t last_update_ms = 0;
};


class RobStrideCan {

// Methods used to send and receive data to / from motors
// =======================================================================================================
public:
  static const uint8_t TOTAL_MOTORS = 12;

  // Default gains (Need to match Isaac Lab actuator config)
  float default_kp = 25.0f;
  float default_kd = 0.8f;

  void setup() {
    can1.begin();
    can1.setBaudRate(1000000);
    can1.onReceive(handleCan1);

    can2.begin();
    can2.setBaudRate(1000000);
    can2.onReceive(handleCan2);
  }


  // Receive incoming CAN messages from motors (feedback)
  void update() {
    can1.events();
    can2.events();
  }


  // Enable / Disable motors
  void enableAll() {
    for (uint8_t id = 1; id <= TOTAL_MOTORS; id++) setMode(id, 0x03);
  }
  void disableAll() {
    for (uint8_t id = 1; id <= TOTAL_MOTORS; id++) setMode(id, 0x04);
  }


  // Give position command to a motor (desired pos, with default PD gains)
  void setTargetPosition(uint8_t id, float desired_pos, float desired_vel = 0.0f, float ff_torque = 0.0f) {
    if (id < 1 || id > TOTAL_MOTORS) return;

    CAN_message_t msg;
    msg.flags.extended = 1;
    msg.len = 8;

    uint16_t t_int  = pack(ff_torque,    T_MIN,  T_MAX);
    uint16_t p_int  = pack(desired_pos,  P_MIN,  P_MAX);
    uint16_t v_int  = pack(desired_vel,  V_MIN,  V_MAX);
    uint16_t kp_int = pack(default_kp,   KP_MIN, KP_MAX);
    uint16_t kd_int = pack(default_kd,   KD_MIN, KD_MAX);

    msg.id = (0x01 << 24) | ((uint32_t)t_int << 8) | id;
    msg.buf[0] = p_int >> 8;  msg.buf[1] = p_int;
    msg.buf[2] = v_int >> 8;  msg.buf[3] = v_int;
    msg.buf[4] = kp_int >> 8; msg.buf[5] = kp_int;
    msg.buf[6] = kd_int >> 8; msg.buf[7] = kd_int;

    write(id, msg);
  }


  // Get feedback from the motors
  const MotorFeedback& getFeedback(uint8_t id) const {
    return feedback[constrain(id, 1, TOTAL_MOTORS) - 1];
  }


  // Check if any motor lost connection (timeout after 200 ms)
  bool isHealthy() const {
    uint32_t now = millis();
    for (uint8_t i = 0; i < TOTAL_MOTORS; i++) {
      if (now - feedback[i].last_update_ms > 200) return false;
    }
    return true;
  }
// =======================================================================================================




// Helpers for this code
// =======================================================================================================
private:
  // CAN BUS & Motor layout
  static FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_32> RobStrideCan::can1;
  static FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_32> RobStrideCan::can2;
  static MotorFeedback RobStrideCan::feedback[RobStrideCan::TOTAL_MOTORS];

  // Set the mode of the motor (enable, disable)
  void setMode(uint8_t id, uint8_t type) {
    CAN_message_t msg;
    msg.flags.extended = 1;
    msg.id = (type << 24) | (0x00FD << 8) | id;
    msg.len = 8;
    memset(msg.buf, 0, 8);
    write(id, msg);
  }


  // Write to the correct CAN bus based on motor ID
  void write(uint8_t id, const CAN_message_t &msg) {
    (id <= 6) ? can1.write(msg) : can2.write(msg);
  }


  // Pack a float in the range [min_v, max_v] to a 16-bit integer
  static uint16_t pack(float val, float min_v, float max_v) {
    val = constrain(val, min_v, max_v);
    return (uint16_t)((val - min_v) * 65535.0f / (max_v - min_v));
  }


  // Unpack a 16-bit integer to a float in the range [min_v, max_v]
  static float unpack(uint16_t raw, float min_v, float max_v) {
    return ((float)raw / 65535.0f) * (max_v - min_v) + min_v;
  }


  // Convert to correct format
  static void parseMessage(const CAN_message_t &msg) {
    uint8_t type = (msg.id >> 24) & 0x1F;
    if (type != 0x02) return; // Ignore non-telemetry frames

    uint8_t id = (msg.id >> 8) & 0xFF;
    if (id < 1 || id > TOTAL_MOTORS) return;

    auto &fb = feedback[id - 1];
    fb.position = unpack((msg.buf[0] << 8) | msg.buf[1], P_MIN, P_MAX);
    fb.velocity = unpack((msg.buf[2] << 8) | msg.buf[3], V_MIN, V_MAX);
    fb.torque   = unpack((msg.buf[4] << 8) | msg.buf[5], T_MIN, T_MAX);
    fb.temp     = ((int16_t)((msg.buf[6] << 8) | msg.buf[7])) * 0.1f;
    fb.last_update_ms = millis();
  }


  // Callback functions from Can bus
  static void handleCan1(const CAN_message_t &msg) { parseMessage(msg); }
  static void handleCan2(const CAN_message_t &msg) { parseMessage(msg); }
// =======================================================================================================
};