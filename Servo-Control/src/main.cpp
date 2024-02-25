#include <Arduino.h>
#include <Servo.h>

#define SERVO_X_IDLE_ANGLE 81
#define SERVO_Y_IDLE_ANGLE 85

#define SERVO_MIN_ANGLE 70
#define SERVO_MAX_ANGLE 110

#define CHECK_SERVO_ANGLE(angle) (SERVO_MIN_ANGLE <= angle && angle <= SERVO_MAX_ANGLE)


Servo servo_x;
Servo servo_y;

void setup() {
    servo_x.attach(11);
    servo_y.attach(10);

    servo_x.write(SERVO_X_IDLE_ANGLE);
    servo_y.write(SERVO_Y_IDLE_ANGLE);

    Serial.begin(115200);
}

void loop() {
    if (Serial.available()) {
        String servoControls = Serial.readStringUntil(';');
        int xy_separator_position = servoControls.indexOf(',');
        int x_val = (int) servoControls.substring(0, xy_separator_position).toInt();
        int y_val = (int) servoControls.substring(xy_separator_position + 1).toInt();

        if (CHECK_SERVO_ANGLE(x_val) && CHECK_SERVO_ANGLE(y_val)) {
            servo_x.write(x_val);
            servo_y.write(y_val);
            Serial.println("Set servo angles");
        } else {
            Serial.println("ERROR: Angle out of range");
        }
    }
}