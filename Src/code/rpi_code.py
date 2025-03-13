import machine, utime, array, sys
import time

SAMPLE_RATE = 100_000
NUM_SAMPLES = 3_000
samples = array.array('H', [0] * NUM_SAMPLES)
sample_index = 0

def sample_callback(timer):
    global sample_index
    global samples
    
    if sample_index < NUM_SAMPLES:
        samples[sample_index] = adc.read_u16()
        sample_index += 1
    else:
        timer.deinit()

def acquire():
    global samples
    global sample_index
    
    sample_index = 0
    
    timer = machine.Timer()
    timer.init(freq=SAMPLE_RATE, mode=machine.Timer.PERIODIC, callback=sample_callback)

    while sample_index < NUM_SAMPLES:
        pass 
    
    print("Sampling complete. Sending data...")

    data_bytes = bytes([val & 0xFF for val in samples] + [(val >> 8) & 0xFF for val in samples])

    chunk_size = 256
    for i in range(0, len(data_bytes), chunk_size):
        uart.write(data_bytes[i:i+chunk_size])
        time.sleep(0.01)

    print("Data transmission complete.")
    return data_bytes

if __name__ == "__main__":
    adc = machine.ADC(26)
    uart = machine.UART(0, baudrate=115200, tx=machine.Pin(12), rx=machine.Pin(13))
    print("UART0 is ready. Waiting for data...")

    while True:
        if uart.any():
            print("Command received!")
            data = uart.read()
            try:
                decoded_data = data.decode('utf-8').strip()
                print("Received command: " + decoded_data)
                if decoded_data == "acquire":
                    print("Acquiring!")
                    acquire()
            except UnicodeError:
                print("Received (raw bytes):", data)
        else:
            print("No information received!")
        time.sleep(1)
