
#define SERVO_MIN_ANGLE 70
#define SERVO_MAX_ANGLE 110

#define CHECK_SERVO_ANGLE(angle) (SERVO_MIN_ANGLE <= angle && angle <= SERVO_MAX_ANGLE)



void setup() {
  Serial.begin(115200);


  pinMode(22, OUTPUT);
  pinMode(23, OUTPUT);
  pinMode(24, OUTPUT);
  pinMode(25, OUTPUT);
  pinMode(26, OUTPUT);
  pinMode(27, OUTPUT);
  pinMode(28, OUTPUT);
  pinMode(29, OUTPUT);

  while (!Serial.available()) {}
  delay(1000);
  if (Serial.available()) {
    String servoControls = Serial.readStringUntil(';');
    int xy_separator_position = servoControls.indexOf(',');
    int x_val = (int)servoControls.substring(0, xy_separator_position).toInt();
    int y_val = (int)servoControls.substring(xy_separator_position + 1).toInt();
  }
}

void setLeds(int val) {
  int p = val ^ (val >> 8);
  p = p ^ (p >> 4);
  p = p ^ (p >> 2);
  p = p ^ (p >> 1);
  p = p & 0x1;


  int b1 = val & 0x7F;
  b1 = b1 | (p << 7);
  b1 = ~b1;

  PORTA = PORTA & ~0xFF;
  PORTA = PORTA | b1;
}



void loop() {

  long start_time = millis();
  for (int i = 0; i < 128; i += 4) {
    setLeds(i);
    while (millis() - start_time < i + 4) {
      // wait
      if (Serial.available()) {
        String servoControls = Serial.readStringUntil(';');
        int xy_separator_position = servoControls.indexOf(',');
        int x_val = (int)servoControls.substring(0, xy_separator_position).toInt();
        int y_val = (int)servoControls.substring(xy_separator_position + 1).toInt();

        if (CHECK_SERVO_ANGLE(x_val) && CHECK_SERVO_ANGLE(y_val)) {
          Serial.println("Set servo angles");
        } else {
          Serial.print("ERROR: Angle out of range: ");
          Serial.print(x_val);
          Serial.print(", ");
          Serial.println(y_val);
        }
        while (true) {
          ;
        }
      }
    }
  }
}

/// END ///
