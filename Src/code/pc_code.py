import serial
import struct
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

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

def find_next_filename(vowel):
    index = 0
    while os.path.exists(f"{vowel}_{index}.txt"):
        index += 1
    return f"{vowel}_{index}.txt"

def save_samples(samples):
    vowel = input("Enter vowel to save (a, e, i, o, u, ha, ui): ").strip().lower()
    if vowel not in ["a", "e", "i", "o", "u", "ha", "ui"]:
        print("Invalid vowel. Try again.")
        return
    
    filename = find_next_filename(vowel)
    with open(filename, "w") as file:
        for value in samples:
            file.write(f"{value}\n")

def load_vowel_data(vowel):

    filenames = glob.glob(f"{vowel}_*.txt")
    if not filenames:
        print(f"No recorded samples found for vowel '{vowel}'.")
        return []

    all_samples = []
    for file in filenames:
        with open(file, "r") as f:
            all_samples.append([int(line.strip()) for line in f.readlines()])
    
    return all_samples

def find_peaks(signal, threshold=1000):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append((i, signal[i]))
    return peaks

def compare_signals(test_signal, recorded_signals):
    if not recorded_signals:
        return float('inf')

    best_score = float('inf')

    for recorded_signal in recorded_signals:
        test_peaks = [p[1] for p in find_peaks(test_signal)]
        recorded_peaks = [p[1] for p in find_peaks(recorded_signal)]

        if not test_peaks or not recorded_peaks:
            continue 

        min_length = min(len(test_peaks), len(recorded_peaks))
        test_peaks = test_peaks[:min_length]
        recorded_peaks = recorded_peaks[:min_length]

        score = np.mean((np.array(test_peaks) - np.array(recorded_peaks)) ** 2)

        if score < best_score:
            best_score = score

    return best_score

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
