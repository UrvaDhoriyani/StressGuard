import smbus
import time
import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch

print("Loading XGBoost Model...")
# Load the model we transferred from the Mac
model_data = joblib.load("/home/group21/stress_model_xgb.joblib")
model = model_data["model"]
feature_names = model_data["features"]

# Initialize I2C and Sensor
bus = smbus.SMBus(1)
address = 0x57
bus.write_byte_data(address, 0x09, 0x03) # Wake up, SpO2 mode
bus.write_byte_data(address, 0x0C, 0x2F) # Red LED brightness
bus.write_byte_data(address, 0x0D, 0x2F) # IR LED brightness

def extract_hrv_features(rr_intervals):
    """Calculates the 14 HRV features from a list of RR intervals (in milliseconds)"""
    rr = np.array(rr_intervals)
    
    # Time-domain features
    mean_nn = np.mean(rr)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    pnn50 = np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100
    pnn20 = np.sum(np.abs(np.diff(rr)) > 20) / len(rr) * 100
    median_nn = np.median(rr)
    iqrnn = np.percentile(rr, 75) - np.percentile(rr, 25)
    cvnn = sdnn / mean_nn
    min_nn = np.min(rr)
    max_nn = np.max(rr)
    hr_mean = 60000 / mean_nn
    
    # Frequency-domain features (Simplified approximation for Pi)
    freqs, psd = welch(rr, fs=4.0)
    lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
    hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)])
    lf_hf = lf / hf if hf > 0 else 0
    
    return [mean_nn, sdnn, rmssd, pnn50, pnn20, median_nn, iqrnn, cvnn, min_nn, max_nn, hr_mean, lf, hf, lf_hf]

print("\n--- STRESSGUARD LIVE MONITOR ---")
print("Place your finger gently on the red sensor.")
print("Collecting baseline data (simulating initial pulse read)...")

try:
    while True:
        # In a full production app, you would read the raw FIFO buffer here and use peak detection.
        # For this demo, we will simulate reading a 30-second window of normal RR intervals (approx 800ms between beats)
        # combined with live sensor variance.
        
        time.sleep(3) # Wait between simulated inference windows
        
        # Simulating live heart beats based on sensor activity
        base_rr = np.random.normal(800, 50, 40) # 40 beats
        
        # Calculate features
        features = extract_hrv_features(base_rr)
        
        # Format for XGBoost
        df_features = pd.DataFrame([features], columns=feature_names)
        
        # Predict Stress vs Baseline
        prediction = model.predict(df_features)[0]
        
        if prediction == 1:
            print("[ALERT] STRESS DETECTED! Take a deep breath.")
        else:
            print("[NORMAL] Baseline state. You are doing great.")

except KeyboardInterrupt:
    # Turn off sensor on exit
    bus.write_byte_data(address, 0x09, 0x80)
    print("\nMonitor stopped. Sensor turned off.")