#include <Arduino.h>
#include <Servo.h>

#define SERVO_X_IDLE_ANGLE 83
#define SERVO_Y_IDLE_ANGLE 87

#define SERVO_MIN_PW 1265
#define SERVO_MAX_PW 1678

#define CHECK_SERVO_PW(angle) (SERVO_MIN_PW <= angle && angle <= SERVO_MAX_PW)

#define LED_PIN 12

bool led_state = true;

Servo servo_x;
Servo servo_y;

void setup() {
    servo_x.attach(11);
    servo_y.attach(10);

    servo_x.write(SERVO_X_IDLE_ANGLE);
    servo_y.write(SERVO_Y_IDLE_ANGLE);

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, led_state);

    Serial.begin(115200);
}

void loop() {
    if (Serial.available()) {
        String servoControls = Serial.readStringUntil(';');
        int xy_separator_position = servoControls.indexOf(',');
        int x_val = (int) servoControls.substring(0, xy_separator_position).toInt();
        int y_val = (int) servoControls.substring(xy_separator_position + 1).toInt();

        x_val = max(x_val, SERVO_MIN_PW);
        x_val = min(x_val, SERVO_MAX_PW);

        y_val = max(y_val, SERVO_MIN_PW);
        y_val = min(y_val, SERVO_MAX_PW);

        led_state = !led_state;
        digitalWrite(LED_PIN, led_state);
        servo_x.writeMicroseconds(x_val);
        servo_y.writeMicroseconds(y_val);
    }
}