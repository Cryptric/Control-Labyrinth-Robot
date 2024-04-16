#include <Arduino.h>
#include <Servo.h>

#define SERVO_X_IDLE_ANGLE 82
#define SERVO_Y_IDLE_ANGLE 86

#define SERVO_MIN_PW 1265
#define SERVO_MAX_PW 1678

#define CHECK_SERVO_PW(angle) (SERVO_MIN_PW <= angle && angle <= SERVO_MAX_PW)


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

        if (CHECK_SERVO_PW(x_val) && CHECK_SERVO_PW(y_val)) {
            servo_x.writeMicroseconds(x_val);
            servo_y.writeMicroseconds(y_val);
            Serial.println("Set servo angles");
        } else {
            Serial.print("ERROR: Angle out of range: ");
            Serial.print(x_val);
            Serial.print(", ");
            Serial.println(y_val);
        }
    }
}