#include <Arduino.h>
#include <Wire.h>
#include <Servo.h>


#define SERVO_X_IDLE_ANGLE 83
#define SERVO_Y_IDLE_ANGLE 87

#define SERVO_MAX_DISPLACEMENT 10

#define CHECK_SERVO_ANGLE(angle) (SERVO_MIN_ANGLE <= angle && angle <= SERVO_MAX_ANGLE)

#define SERVO_WAIT_TIME 1000

const int MPU6500Address = 0x68; // MPU-6500 I2C address

Servo servo_x;
Servo servo_y;

double reference_vector[3];


double calc_norm(double* vec) {
    return sqrt(sq(vec[0]) + sq(vec[1]) + sq(vec[2]));
}

double calc_dot(double* a, double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double get_angle_sign(double* a, double* b) {
    double cross_z = a[0] * b[1] - a[1] * b[0];
    if (cross_z < 0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

/*
    Calculate the angle between two vectors
*/
double calc_angle(double* a, double* b) {
    return get_angle_sign(a, b) * acos(calc_dot(a, b) / (calc_norm(a) * calc_norm(b))) / PI * 180;
}

void initializeMPU6500() {
    // Power management register (Register 107)
    Wire.beginTransmission(MPU6500Address);
    Wire.write(0x6B); // PWR_MGMT_1 register
    Wire.write(0);    // Set to 0 (wakes up the MPU-6500)
    Wire.endTransmission(true);

    // Set the accelerometer range (Register 28)
    Wire.beginTransmission(MPU6500Address);
    Wire.write(0x1C); // ACCEL_CONFIG register
    Wire.write(0x10); // Set to 0x10 for +/- 8g range (adjust as needed)
    Wire.endTransmission(true);
}

void readAccelerometer(double* accelX, double* accelY, double* accelZ, int n) {
    int32_t accX = 0;
    int32_t accY = 0;
    int32_t accZ = 0;
    for (int i = 0; i < n; i++) {
        Wire.beginTransmission(MPU6500Address);
        Wire.write(0x3B); // Starting register for accelerometer data
        Wire.endTransmission(false);
        Wire.requestFrom(MPU6500Address, 6, true);
        accX += Wire.read() << 8 | Wire.read();
        accY += Wire.read() << 8 | Wire.read();
        accZ += Wire.read() << 8 | Wire.read();
        delay(50);
    }

    *accelX = accX / ((float) n);
    *accelY = accY / ((float) n);
    *accelZ = accZ / ((float) n);
}

void perform_measurement(Servo& servo, int alpha, int idle_position) {
    servo.write(alpha);
    delay(SERVO_WAIT_TIME);
    double accel[3];
    readAccelerometer(&accel[0], &accel[1], &accel[2], 1);
    double tilt = calc_angle(reference_vector, accel);
    Serial.print(alpha - idle_position);
    Serial.print(",");
    Serial.println(tilt);
}

void perform_routine(Servo& servo, int idle_position) {
    servo.write(idle_position);
    int alpha = idle_position + SERVO_MAX_DISPLACEMENT;
    for (; alpha >= idle_position - SERVO_MAX_DISPLACEMENT; alpha--) {
        perform_measurement(servo, alpha, idle_position);
    }
    servo.write(idle_position);
    delay(SERVO_WAIT_TIME);
    alpha = idle_position - SERVO_MAX_DISPLACEMENT;
    for (; alpha <= idle_position + SERVO_MAX_DISPLACEMENT; alpha++) {
        perform_measurement(servo, alpha, idle_position);
    }
}

void setup() {
    Serial.begin(115200);
    Wire.begin();

    // Initialize MPU-6500
    initializeMPU6500();

    servo_x.attach(11);
    servo_y.attach(10);

    servo_x.write(SERVO_X_IDLE_ANGLE);
    servo_y.write(SERVO_Y_IDLE_ANGLE);

    delay(SERVO_WAIT_TIME);
    readAccelerometer(reference_vector, reference_vector + 1, reference_vector + 2, 1);

    delay(500);

    Serial.println("Measure Servo X");
    perform_routine(servo_x, SERVO_X_IDLE_ANGLE);

    servo_x.write(SERVO_X_IDLE_ANGLE);
    delay(2 * SERVO_WAIT_TIME);
    readAccelerometer(reference_vector, reference_vector + 1, reference_vector + 2, 10);
    delay(SERVO_WAIT_TIME);

    
    Serial.println("Measure Servo Y");
    perform_routine(servo_y, SERVO_Y_IDLE_ANGLE);
    servo_y.write(SERVO_Y_IDLE_ANGLE);
}


void loop() {

}
