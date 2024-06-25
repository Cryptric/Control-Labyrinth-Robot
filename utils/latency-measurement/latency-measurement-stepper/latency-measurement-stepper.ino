
#define SERVO_MIN_PW 1265
#define SERVO_MAX_PW 1678

#define CHECK_SERVO_PW(angle) (SERVO_MIN_PW <= angle && angle <= SERVO_MAX_PW)

#define STEP_PIN 7
#define DIR_PIN 6

void setup() {
  // put your setup code here, to run once:

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    String servoControls = Serial.readStringUntil(';');
    int xy_separator_position = servoControls.indexOf(',');
    int x_val = (int)servoControls.substring(0, xy_separator_position).toInt();
    int y_val = (int)servoControls.substring(xy_separator_position + 1).toInt();

    if (CHECK_SERVO_PW(x_val) && CHECK_SERVO_PW(y_val)) {
      int sleep = 600;
      for (int i = 0; i < 25; i++) {
        digitalWrite(STEP_PIN, HIGH);
        delayMicroseconds(50);
        digitalWrite(STEP_PIN, LOW);
        delayMicroseconds(sleep);
        sleep = max(50, sleep - 16);
      }
      Serial.println("Set servo angles");
    } else {
      Serial.print("ERROR: Angle out of range: ");
      Serial.print(x_val);
      Serial.print(", ");
      Serial.println(y_val);
    }
  }
}
