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

def plot_signals(original, matched, vowel):
    original_fft = np.fft.fft(original)
    matched_fft = np.fft.fft(matched)
    freqs = np.fft.fftfreq(len(original))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(original, label="Acquired Signal", color='b')
    plt.plot(matched, label=f"Matched Vowel ({vowel})", color='r', linestyle='dashed')
    plt.xlabel("Sample Index")
    plt.ylabel("ADC Value")
    plt.title(f"Time Domain Comparison - Acquired vs. {vowel.upper()}")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(freqs[:len(freqs)//2], np.abs(original_fft[:len(freqs)//2]), label="Acquired Spectrum", color='b')
    plt.plot(freqs[:len(freqs)//2], np.abs(matched_fft[:len(freqs)//2]), label=f"Matched Spectrum ({vowel})", color='r', linestyle='dashed')
    plt.xlabel("Frequency (Normalized)")
    plt.ylabel("Magnitude")
    plt.title(f"Frequency Spectrum Comparison - Acquired vs. {vowel.upper()}")
    plt.legend()
    plt.grid()

    plt.tight_layout()
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
    best_match = None
    
    for recorded_signal in recorded_signals:
        test_peaks = [p[1] for p in find_peaks(test_signal, 5000)]
        recorded_peaks = [p[1] for p in find_peaks(recorded_signal,5000)]

        if not test_peaks or not recorded_peaks:
            continue 

        min_length = min(len(test_peaks), len(recorded_peaks))
        test_peaks = test_peaks[:min_length]
        recorded_peaks = recorded_peaks[:min_length]

        score = np.mean((np.array(test_peaks) - np.array(recorded_peaks)) ** 2)

        if score < best_score:
            best_score = score
            best_match = recorded_signal

    return best_score, best_match

def compute_fft(signal):
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    
    magnitude = np.abs(fft_values)
    sorted_indices = np.argsort(magnitude)[::-1]

    return freqs[sorted_indices[1]], freqs[sorted_indices[2]]

def compare_signals_fft(test_signal, recorded_signals):
    if not recorded_signals:
        return float('inf')

    best_score = float('inf')
    best_match = None

    test_f2, test_f3 = compute_fft(test_signal)
    for recorded_signal in recorded_signals:
        rec_f2, rec_f3 = compute_fft(recorded_signal)
        score = abs(test_f2 - rec_f2) + abs(test_f3 - rec_f3)

        if score < best_score:
            best_score = score
            best_match = recorded_signal

    return best_score, best_match

def test_vowel(samples):
    vowels = ["a", "e", "i", "o", "u", "ha", "ui"]
    best_match_time = None
    best_match_freq = None
    best_score_time = float('inf')
    best_score_freq = float('inf')

    for vowel in vowels:
        recorded_signals = load_vowel_data(vowel)
        if recorded_signals is None:
            continue
        
        freq_score, freq_best_signal = compare_signals_fft(samples, recorded_signals)
        print(f"FREQ Comparison score with '{vowel}': {freq_score}")

        time_score, time_best_signal = compare_signals(samples, recorded_signals)
        print(f"TIME Comparison score with '{vowel}': {time_score}")

        if time_score < best_score_time:
            best_score_time = time_score
            best_match_time = vowel

        if freq_score < best_score_freq:
            best_score_freq = freq_score
            best_match_freq = vowel

    print(f"Closest match in time: {best_match_time.upper()}")
    print(f"Closest match in freq: {best_match_freq.upper()}")

    plot_signals(samples, time_best_signal, best_match_time)
    plot_signals(samples, freq_best_signal, best_match_freq)




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

            inpt = input("Continue? 1 - Yes, 0 - No\n")
            if("0" == inpt):
                break
