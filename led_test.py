import smbus
import time

# Initialize I2C bus
bus = smbus.SMBus(1)
address = 0x57

try:
    print("Initializing MAX30102...")
    
    # 0x09 is the Mode Configuration Register. 
    # Sending 0x03 puts it in SpO2 mode (turns on Red and IR LEDs)
    bus.write_byte_data(address, 0x09, 0x03) 
    
    # 0x0C is the LED1 (Red) Pulse Amplitude Register. 
    # Sending 0x2F sets the brightness to a visible level.
    bus.write_byte_data(address, 0x0C, 0x2F) 
    
    # 0x0D is the LED2 (IR) Pulse Amplitude Register.
    bus.write_byte_data(address, 0x0D, 0x2F) 
    
    print("SUCCESS! The red LED inside the sensor should now be ON.")
    print("Press Ctrl+C to turn it off and exit.")
    
    # Keep the script running so the light stays on
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    # When you press Ctrl+C, send the reset command to turn the LED back off
    bus.write_byte_data(address, 0x09, 0x80)
    print("\nSensor turned off. Goodbye!")
except Exception as e:
    print(f"An error occurred: {e}")