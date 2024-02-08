#include <Arduino.h>
#include <Servo.h>

#define SERVO_MIN_ANGLE 50
#define SERVO_MAX_ANGLE 130

#define CHECK_SERVO_ANGLE(angle) (SERVO_MIN_ANGLE <= angle && angle <= SERVO_MAX_ANGLE)


Servo servo_x;
Servo servo_y;

void setup() {
    servo_x.attach(11);
    servo_y.attach(10);

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
        } else {
            Serial.println("ERROR: Angle out of range");
        }
    }
}