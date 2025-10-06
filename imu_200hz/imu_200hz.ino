#include <Wire.h>
#include <LSM6DSOX.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_Sensor.h>

Adafruit_LIS3MDL mag;

void setup(void) {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("Initializing sensors...");

  // 初始化 LSM6DSOX
  if (!IMU.begin()) {
    Serial.println("Failed to find LSM6DSOX!");
    while (1) delay(10);
  }
  Serial.println("LSM6DSOX ready.");

  // 初始化 LIS3MDL（磁力计）
  if (!mag.begin_I2C()) {
    Serial.println("Failed to find LIS3MDL!");
    while (1) delay(10);
  }
  Serial.println("LIS3MDL ready.");

  // 配置磁力计参数
  mag.setPerformanceMode(LIS3MDL_ULTRAHIGHMODE);
  mag.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  mag.setDataRate(LIS3MDL_DATARATE_155_HZ);
  mag.setRange(LIS3MDL_RANGE_4_GAUSS);
}

unsigned long prevMicros = 0;
const unsigned long interval = 5000; // 200 Hz → 1/200 s = 5000 μs

void loop() {
  unsigned long now = micros();
  if (now - prevMicros >= interval) {
    prevMicros = now;

    // --- IMU 数据 ---
    float ax, ay, az;
    float gx, gy, gz;
    if (IMU.accelerationAvailable()) IMU.readAcceleration(ax, ay, az);
    if (IMU.gyroscopeAvailable()) IMU.readGyroscope(gx, gy, gz);

    // --- 磁力计数据 ---
    sensors_event_t mag_event;
    mag.getEvent(&mag_event);

    // --- CSV 输出 ---
    Serial.print(ax,3); Serial.print(",");
    Serial.print(ay,3); Serial.print(",");
    Serial.print(az,3); Serial.print(",");
    Serial.print(gx,3); Serial.print(",");
    Serial.print(gy,3); Serial.print(",");
    Serial.print(gz,3); Serial.print(",");
    Serial.print(mag_event.magnetic.x,2); Serial.print(",");
    Serial.print(mag_event.magnetic.y,2); Serial.print(",");
    Serial.println(mag_event.magnetic.z,2);
  }
}
