# 🎤 Vowel Recognition Using ADC and FFT

This project implements **real-time vowel recognition** using **ADC signal acquisition, Fast Fourier Transform (FFT) analysis, and peak detection**. It compares **acquired audio signals** with **pre-recorded vowels**, identifies the best match, and plots both the **time-domain waveform** and **frequency spectrum** for comparison.

---

## 📌 **Features**
✅ **Real-time ADC Sampling via UART**  
✅ **Save multiple training vowel recordings (`a_0.txt, e_0.txt`, etc.)**  
✅ **Identify vowels based on FFT frequency peaks**  
✅ **Plots both waveform & spectrum of matched vowel vs. acquired signal**  
✅ **Supports multiple recordings per vowel for improved accuracy**  
✅ **Two different comparison methods (Time-Domain & Frequency-Domain)**

---

## ⚙️ **Hardware & Software Requirements**

### **🔹 Hardware**
- **Raspberry Pi Pico** (or compatible microcontroller with ADC)
- **Electret Microphone** or **Analog Sound Sensor**
- **PL2303TA TTL USB to UART Cable**
- **Windows/Linux/MacOS** with a **USB-UART connection**

### **🔹 Software**
- **Python 3.7+**
- **MicroPython** installed on the Raspberry Pi Pico
- Required Python libraries:
  ```sh
  pip install pyserial numpy matplotlib
  ```

---

## 🚀 **Setup & Usage**

### **1️⃣ Microcontroller - Signal Acquisition**
1. **Flash MicroPython** onto the Raspberry Pi Pico.
2. **Run the MicroPython script** on the Pico using Thonny.
3. **Ensure the correct UART TX/RX pins are configured.**

### **2️⃣ Python - Run the Main Script**
```sh
python main.py
```

### **3️⃣ Save Training Data**
1. **Choose `0`** after acquiring a vowel.
2. **Enter the vowel name (`a, e, i, o, u, ha, ui`).**
3. The script **automatically saves multiple recordings** (`a_0.txt`, `a_1.txt`, etc.).

### **4️⃣ Test Vowel Recognition**
1. **Choose `1`** after acquiring an unknown vowel.
2. The program:
   - **Compares FFT frequency peaks** with saved recordings.
   - **Finds the closest matching vowel** using both Time & Frequency comparison.
   - **Plots time & frequency spectrum comparison.**

---

**Graph Output:**
- **Time-Domain:** Acquired waveform (blue) vs. Matched vowel (red, dashed).
- **Frequency Spectrum:** FFT spectrum of both signals.


---

## 🔧 **Customization**
- **Modify `NUM_SAMPLES`** in `main.py` for higher/lower resolution.
- **Change `COM_PORT`** to match your UART connection.
- **Tune `threshold` in `find_peaks()`** to adjust peak sensitivity.
- **Use different sensors** (e.g., MEMS microphone) for better signal capture.

---

## 🛠️ **Future Improvements**
- 📈 **Add Spectrogram Analysis** for better frequency representation.
- 🎵 **Train Machine Learning Model** to classify vowels.
- 🔀 **Use Dynamic Time Warping (DTW)** for better pattern matching.
- 📡 **Implement Wireless UART (ESP-NOW, Bluetooth, etc.).**
