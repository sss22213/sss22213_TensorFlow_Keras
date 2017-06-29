#include <SerialLCD.h>
#include <SoftwareSerial.h>
const char *senstens = "Weclome to MCUEE";
SerialLCD slcd(11, 12);
void setup()
{
  slcd.begin();
  delay(1000);
  slcd.setCursor(0, 0);
  slcd.print(senstens);
}
void loop()
{

}

