const int IR_PIN = 2; 
int lastState = HIGH;

void setup() {
  Serial.begin(9600);
  pinMode(IR_PIN, INPUT);
}

void loop() {
  int currentState = digitalRead(IR_PIN);
  
  // IR sensor gives LOW when object is detected
  if (currentState == LOW && lastState == HIGH) {
    Serial.println("DETECTED");
    delay(500); 
  }
  
  lastState = currentState;
}