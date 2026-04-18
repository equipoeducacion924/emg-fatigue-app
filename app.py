import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch

# --- CONFIGURATION AND FUNCTIONS ---
st.set_page_config(page_title="Advanced EMG Fatigue Dashboard", layout="wide")

def bandpass_filter(x, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    return filtfilt(b, a, x)

def notch_filter(x, fs):
    nyq = 0.5 * fs
    w0 = 60/nyq # Assuming 60Hz, adjust if your region uses 50Hz
    b, a = iirnotch(w0, 30)
    return filtfilt(b, a, x)

def get_metrics(segment, fs):
    f, p = welch(segment, fs=fs, nperseg=len(segment))
    power = p
    freqs = f
    mnf = np.sum(freqs * power) / np.sum(power) if np.sum(power) > 0 else 0
    cum_power = np.cumsum(power)
    total_p = cum_power[-1] if len(cum_power) > 0 else 0
    mdf = freqs[np.where(cum_power >= total_p/2)[0][0]] if total_p > 0 else 0
    rms = np.sqrt(np.mean(segment**2))
    return mnf, mdf, rms, freqs, power

# --- INTERFACE ---
st.title("⚡ Advanced EMG Fatigue Dashboard")

archivo = st.file_uploader("Upload your processed CSV file", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    columna = st.selectbox("Select the signal", df.columns)
    fs = st.number_input("Sampling Rate (Hz)", value=1000)
    
    sig_raw = df[columna].values
    sig_f = notch_filter(bandpass_filter(sig_raw, fs), fs)
    
    # Visualization Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Signal (Raw vs Clean)", "SpectrUM (PSD)", "Temporal evolution", "Fatigue Metrics"])
    
    # Tab 1: Comparison
    with tab1:
        st.subheader("Signal Comparison")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(sig_raw[:1000], label="Raw", alpha=0.5)
        ax1.plot(sig_f[:1000], label="Filtered", color='green')
        ax1.legend()
        st.pyplot(fig1)
        
    # Tab 2: Spectrum
    with tab2:
        st.subheader("Power Spectral Density (PSD)")
        f, p = welch(sig_f, fs=fs)
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.semilogy(f, p)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("PSD")
        st.pyplot(fig2)
        
    # Tab 3 & 4: Fatigue Calculation
    win_sec = 0.5
    w_len = int(win_sec * fs)
    segs = [sig_f[i:i+w_len] for i in range(0, len(sig_f)-w_len, w_len)]
    metrics = [get_metrics(seg, fs) for seg in segs]
    mnf_v, mdf_v, rms_v, _, _ = zip(*metrics)
    t_segs = np.arange(len(segs)) * win_sec

    with tab3:
        st.subheader("Frequency and Amplitude Evolution")
        fig3, ax3 = plt.subplots(2, 1, figsize=(10, 6))
        ax3[0].plot(t_segs, mnf_v, color='blue', label="MNF")
        ax3[0].plot(t_segs, mdf_v, color='red', label="MDF")
        ax3[0].legend()
        ax3[1].plot(t_segs, rms_v, color='orange', label="RMS")
        ax3[1].legend()
        st.pyplot(fig3)

    with tab4:
        st.subheader("Fatigue State")
        drop = ((mnf_v[-1] - mnf_v[0]) / mnf_v[0]) * 100
        st.metric("MNF Drop (%)", f"{drop:.2f}%")
        if drop < -15:
            st.error("Level: FATIGUE DETECTED")
        else:
            st.success("Level: FRESH MUSCLE")

else:
    st.info("Please upload a CSV file to get started.")
