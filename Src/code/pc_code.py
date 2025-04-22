import serial
import struct
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

# Replace with your actual COM port (Check Device Manager)
COM_PORT = "COM4"  
BAUD_RATE = 115_200  
NUM_SAMPLES = 1_000

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "processed_data"
DATA_DIR.mkdir(exist_ok=True)

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

def plot_signals(original, matched_time, vowel_time, matched_freq, vowel_freq):
    N = len(original)
    orig_fft = np.fft.fft(original)
    match_fft = np.fft.fft(matched_freq)
    freqs = np.fft.fftfreq(N)

    half = N // 2

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(original, label="Acquired Signal", linewidth=1)
    plt.plot(matched_time, linestyle='--', label=f"Matched Vowel ({vowel_time.upper()})", linewidth=1)
    plt.xlabel("Sample Index")
    plt.ylabel("ADC Value")
    plt.title(f"Time Domain Comparison vs {vowel_time.upper()}")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freqs[:half], np.abs(orig_fft[:half]), label="Acquired Spectrum", linewidth=1)
    plt.plot(freqs[:half], np.abs(match_fft[:half]), linestyle='--', label=f"Freq‑Domain Match ({vowel_freq.upper()})", linewidth=1)
    plt.xlabel("Frequency (Normalized)")
    plt.ylabel("Magnitude")
    plt.title(f"Frequency Spectrum Comparison vs {vowel_freq.upper()}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def apply_window(signal):
    x = np.array(signal, dtype=float)
    N = len(x)
    w = np.hamming(N)
    return x * w

def normalize_signal(signal):
    x = np.array(signal, dtype=float)
    x = x - np.mean(x)
    std = np.std(x) + 1e-8
    x = x / std
    return x

def find_next_filename(vowel):
    i = 0
    candidate = DATA_DIR / f"{vowel}_{i}.txt"
    while candidate.exists():
        i += 1
        candidate = DATA_DIR / f"{vowel}_{i}.txt"
    return candidate


def align_signals(test, template):
    corr = np.correlate(test, template, mode="full")
    lag = corr.argmax() - (len(template) - 1)

    if lag > 0:
        template_aligned = np.pad(template, (lag, 0), mode="constant")[:len(test)]
        test_aligned     = test[:len(template_aligned)]
    else:
        test_aligned     = np.pad(test, (-lag, 0), mode="constant")[:len(template)]
        template_aligned = template[:len(test_aligned)]

    return test_aligned, template_aligned

def save_samples(samples):
    vowel = input("Enter vowel to save (a, e, i, o, u, ha, ui): ").strip().lower()
    if vowel not in ["a", "e", "i", "o", "u", "ha", "ui"]:
        print("Invalid vowel. Try again.")
        return
    
    normed = normalize_signal(samples)
    filename = find_next_filename(vowel)
    with open(filename, "w") as file:
        for value in normed:
            file.write(f"{value:.6f}\n")

def load_vowel_data(vowel):
    filenames = sorted(DATA_DIR.glob(f"{vowel}_*.txt"))
    if not filenames:
        print(f"No recorded samples found for vowel '{vowel}'.")
        return []

    all_samples = []
    for file in filenames:
        with open(file, "r") as f:
            all_samples.append([float(line.strip()) for line in f])
    
    return all_samples

def find_peaks(signal, threshold=1000):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append((i, signal[i]))
    return peaks

def compare_signals_mse(test, templates):
    best_score, best_match = float('inf'), None
    for tpl in templates:
        t_al, tpl_al = align_signals(test, tpl)
        score = np.mean((t_al - tpl_al)**2)
        if score < best_score:
            best_score, best_match = score, tpl
    return best_score, best_match


def compare_signals(test_signal, recorded_signals):
    if not recorded_signals:
        return float('inf')

    best_score, best_match = float('inf'), None
    
    for rec in recorded_signals:
        t_aligned, r_aligned = align_signals(test_signal, rec)
        tp = [p[1] for p in find_peaks(t_aligned, 1.5)]
        rp = [p[1] for p in find_peaks(r_aligned, 1.5)]

        if not tp or not rp:
            continue

        n = min(len(tp), len(rp))
        tp, rp = np.array(tp[:n]), np.array(rp[:n])

        score = np.mean((tp - rp) ** 2)

        if score < best_score:
            best_score, best_match = score, rec

    return best_score, best_match

def compute_fft(signal):
    win_sig = apply_window(signal)
    fft_vals = np.fft.fft(win_sig)
    freqs = np.fft.fftfreq(len(win_sig))
    mag = np.abs(fft_vals)
    idx_sorted = np.argsort(mag)[::-1]
    f2 = freqs[ idx_sorted[1] ]
    f3 = freqs[ idx_sorted[2] ]

    return f2, f3

def compare_signals_fft(test_signal, recorded_signals):
    if not recorded_signals:
        return float('inf')

    best_score, best_match = float('inf'), None
    test_f2, test_f3 = compute_fft(test_signal)

    for rec in recorded_signals:
        t_al, r_al = align_signals(test_signal, rec)
        rec_f2, rec_f3 = compute_fft(r_al)
        score = abs(test_f2 - rec_f2) + abs(test_f3 - rec_f3)
        if score < best_score:
            best_score, best_match = score, rec

    return best_score, best_match

def test_vowel(samples):
    vowels = ["a", "e", "i", "o", "u", "ha", "ui"]
    best_match_time = None
    best_match_freq = None
    best_score_time = float('inf')
    best_score_freq = float('inf')

    for vowel in vowels:
        recorded_signals = load_vowel_data(vowel)
        if not recorded_signals:
            continue
        
        freq_score, freq_best_signal = compare_signals_fft(samples, recorded_signals)
        print(f"FREQ Comparison score with '{vowel}': {freq_score}")

        time_score, time_best_signal = compare_signals_mse(samples, recorded_signals)
        print(f"TIME Comparison score with '{vowel}': {time_score}")

        if time_score < best_score_time:
            best_score_time = time_score
            best_match_time = vowel

        if freq_score < best_score_freq:
            best_score_freq = freq_score
            best_match_freq = vowel

    if best_match_time is not None and best_match_freq is not None:
        print(f"Closest match in time: {best_match_time.upper()}")
        print(f"Closest match in freq: {best_match_freq.upper()}")
        plot_signals(samples, time_best_signal, best_match_time, freq_best_signal, best_match_freq)

    else:
        print("No valid match found.")

if __name__ == "__main__":
    inpt = input("Are you ready to start? 1 - Yes, 0 - No\n")
    if "1" == inpt:
        while(True):
            raw = receive_samples()
            samples = normalize_signal(raw) 
            usr_inpt = input("Acquisition succesfully! (0 - Save data, 1 - Test vowel)\n")
            if "0" == usr_inpt:
                save_samples(samples)
            elif "1" == usr_inpt:
                test_vowel(samples)

            inpt = input("Continue? 1 - Yes, 0 - No\n")
            if("0" == inpt):
                break
