import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fft import fft, fftfreq

# --- CONFIGURACIÓN Y FUNCIONES DE PROCESAMIENTO (Tu lógica de Colab) ---
def bandpass_filter(x, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    return filtfilt(b, a, x)

def notch_filter(x, fs):
    nyq = 0.5 * fs
    w0 = 60/nyq
    b, a = iirnotch(w0, 30)
    return filtfilt(b, a, x)

def get_metrics(segment, fs):
    N = len(segment)
    yf = np.abs(fft(segment))[:N//2]
    xf = fftfreq(N, 1/fs)[:N//2]
    power = yf**2
    mnf = np.sum(xf * power) / np.sum(power) if np.sum(power) > 0 else 0
    cum_power = np.cumsum(power)
    total_p = cum_power[-1] if len(cum_power) > 0 else 0
    mdf = xf[np.where(cum_power >= total_p/2)[0][0]] if total_p > 0 else 0
    rms = np.sqrt(np.mean(segment**2))
    return mnf, mdf, rms

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="EMG Fatigue Analyzer", layout="wide")
st.title("⚡ Analizador de Fatiga Muscular (EMG)")

archivo = st.file_uploader("Sube tu señal EMG (CSV)", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    columna = st.selectbox("Selecciona la columna de la señal EMG", df.columns)
    fs = st.number_input("Frecuencia de muestreo (Hz)", value=1000)
    
    # Procesamiento
    sig_raw = df[columna].values
    sig_f = notch_filter(bandpass_filter(sig_raw, fs), fs)
    
    # Segmentación (0.5s)
    win_sec = 0.5
    w_len = int(win_sec * fs)
    segs = [sig_f[i:i+w_len] for i in range(0, len(sig_f)-w_len, w_len)]
    t_segs = np.arange(len(segs)) * win_sec
    
    # Calcular métricas
    metrics = [get_metrics(seg, fs) for seg in segs]
    mnf_v, mdf_v, rms_v = zip(*metrics)
    
    # Baselines (Primeros 10s según tu Colab)
    n_base = max(1, int(10 / win_sec))
    base_mnf = np.mean(mnf_v[:n_base])
    
    # Cálculo de nivel de fatiga actual
    current_mnf_drop = ((mnf_v[-1] - base_mnf) / base_mnf) * 100
    
    # Clasificación de nivel
    def clasificar(diff):
        if diff < -25: return "Severe Fatigue", "🔴"
        elif diff < -15: return "Moderate Fatigue", "🟠"
        elif diff < -5: return "Mild Fatigue", "🟡"
        return "No Fatigue", "🟢"

    nivel, emoji = clasificar(current_mnf_drop)

    # --- DASHBOARD ---
    st.header(f"Estado Actual: {emoji} {nivel}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("MNF Drop", f"{current_mnf_drop:.2f}%")
    c2.metric("MDF Actual", f"{mdf_v[-1]:.2f} Hz")
    c3.metric("RMS Actual", f"{rms_v[-1]:.4f}")

    # Gráficas
    st.subheader("Evolución de Parámetros")
    tab1, tab2 = st.tabs(["Frecuencias (MNF/MDF)", "Amplitud (RMS)"])
    
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t_segs, mnf_v, label="MNF")
        ax1.plot(t_segs, mdf_v, label="MDF")
        ax1.set_ylabel("Frecuencia (Hz)")
        ax1.legend()
        st.pyplot(fig1)
        
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(t_segs, rms_v, color="orange")
        ax2.set_ylabel("RMS (Amplitud)")
        st.pyplot(fig2)

else:
    st.info("Sube un archivo para procesar la fatiga en tiempo real.")
