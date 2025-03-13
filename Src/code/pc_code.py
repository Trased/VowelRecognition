import serial
import struct
import matplotlib.pyplot as plt
import numpy as np
import os

# Replace with your actual COM port (Check Device Manager)
COM_PORT = "COM11"  
BAUD_RATE = 115_200  
NUM_SAMPLES = 1_000

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
            
            print(len(raw_data))
            if len(raw_data) == expected_bytes:
                # Decode the bytes into unsigned 16-bit integers
                samples = list(struct.unpack(f"<{NUM_SAMPLES}H", raw_data))
                return samples[len(samples)//2:]
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

def save_samples(samples):
    filename = input("Enter filename to save data: ")
    with open(filename, "w") as file:
        for value in samples:
            file.write(f"{value}\n")


def load_vowel_data(vowel):
    filename = f"{vowel}.txt"
    if not os.path.exists(filename):
        print(f"⚠️ File '{filename}' not found.")
        return None
    with open(filename, "r") as file:
        return [int(line.strip()) for line in file.readlines()]

def find_peaks(signal, threshold=1000):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append((i, signal[i]))
    return peaks

def compare_signals(test_signal, recorded_signal):
    if recorded_signal is None:
        return float('inf')

    test_peaks = [p[1] for p in find_peaks(test_signal)]
    recorded_peaks = [p[1] for p in find_peaks(recorded_signal)]

    if not test_peaks or not recorded_peaks:
        return float('inf')

    min_length = min(len(test_peaks), len(recorded_peaks))
    test_peaks = test_peaks[:min_length]
    recorded_peaks = recorded_peaks[:min_length]

    similarity = np.mean((np.array(test_peaks) - np.array(recorded_peaks)) ** 2)
    return similarity

def test_vowel(samples):
    vowels = ["a", "e", "i", "o", "u", "ha", "ui"]
    best_match = None
    best_score = float('inf')

    for vowel in vowels:
        recorded_signal = load_vowel_data(vowel)
        if recorded_signal is None:
            continue

        score = compare_signals(samples, recorded_signal)
        print(f"Comparison score with '{vowel}': {score}")

        if score < best_score:
            best_score = score
            best_match = vowel

    if best_match:
        print(f"Closest match: {best_match.upper()}")
    else:
        print("No matching vowel found.")



if __name__ == "__main__":
    inpt = input("Are you ready to start? 1 - Yes, 0 - No\n")
    if "1" == inpt:
        while(True):
            samples = receive_samples()
            usr_inpt = input("Acquisition succesfully! (0 - Save data, 1 - Test vowel)\n")
            if "0" == usr_inpt:
                save_samples(samples)
            elif "1" == usr_inpt:
                test_vowel(samples)
            
            #plot_samples(samples)

            inpt = input("Continue? 1 - Yes, 0 - No\n")
            if("0" == inpt):
                break
