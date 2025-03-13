import serial
import struct
import matplotlib.pyplot as plt

# Replace with your actual COM port (Check Device Manager)
COM_PORT = "COM23"  
BAUD_RATE = 115200  
NUM_SAMPLES = 300 

def receive_samples():
    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=2) as ser:
            print(f"Listening on {COM_PORT} at {BAUD_RATE} baud...")
            command = "acquire\n"
            ser.write(command.encode('utf-8'))
            print("Command sent!")
            
            # Read the expected number of bytes (NUM_SAMPLES * 2 because each sample is 2 bytes)
            expected_bytes = NUM_SAMPLES * 2
            raw_data = ser.read(expected_bytes)
            
            if len(raw_data) == expected_bytes:
                # Decode the bytes into unsigned 16-bit integers
                samples = list(struct.unpack(f"<{NUM_SAMPLES}H", raw_data))
                
                # Print the ADC samples
                print("\nReceived ADC Samples:")
                print(samples)

                return samples
            else:
                print(f"Error: Expected {expected_bytes} bytes, but received {len(raw_data)} bytes.")

    except serial.SerialException as e:
        print(f"Serial error: {e}")

def plot_samples(samples):
    if samples is None:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(samples, label="ADC Readings", color="b")
    plt.xlabel("Sample Index")
    plt.ylabel("ADC Value (0-65535)")
    plt.title("ADC Data Acquisition")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    samples = receive_samples()
    plot_samples(samples)
