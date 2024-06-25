#include <Servo.h>

#define SERVO_X_IDLE_ANGLE 82
#define SERVO_Y_IDLE_ANGLE 86

#define SERVO_MIN_ANGLE 70
#define SERVO_MAX_ANGLE 110

#define CHECK_SERVO_ANGLE(angle) (SERVO_MIN_ANGLE <= angle && angle <= SERVO_MAX_ANGLE)




Servo servo_x;

void setup() {
    servo_x.attach(11);
    
    
    pinMode(9, OUTPUT);
    
    Serial.begin(115200);
}

void loop() {
    servo_x.write(0);
    delay(112);
    digitalWrite(9, HIGH);
    servo_x.write(180);
    delay(112);
    digitalWrite(9, LOW);
}
