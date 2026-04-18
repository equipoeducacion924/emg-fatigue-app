import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch

# --- CONFIGURACIÓN Y FUNCIONES ---
st.set_page_config(page_title="Dashboard EMG Completo", layout="wide")

def bandpass_filter(x, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    return filtfilt(b, a, x)

def notch_filter(x, fs):
    nyq = 0.5 * fs
    w0 = 60/nyq # Asumiendo 60Hz, ajusta si en tu país es 50Hz
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

# --- INTERFAZ ---
st.title("⚡ Dashboard Avanzado de Fatiga EMG")

archivo = st.file_uploader("Sube tu archivo CSV procesado", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    columna = st.selectbox("Selecciona la señal", df.columns)
    fs = st.number_input("Frecuencia de muestreo (Hz)", value=1000)
    
    sig_raw = df[columna].values
    sig_f = notch_filter(bandpass_filter(sig_raw, fs), fs)
    
    # Pestañas de Visualización
    tab1, tab2, tab3, tab4 = st.tabs(["Señal (Raw vs Clean)", "Espectro (PSD)", "Evolución Temporal", "Métricas de Fatiga"])
    
    # Tab 1: Comparativa
    with tab1:
        st.subheader("Comparativa de Señal")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(sig_raw[:1000], label="Raw", alpha=0.5)
        ax1.plot(sig_f[:1000], label="Filtrada", color='green')
        ax1.legend()
        st.pyplot(fig1)
        
    # Tab 2: Espectro
    with tab2:
        st.subheader("Densidad Espectral de Potencia (PSD)")
        f, p = welch(sig_f, fs=fs)
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.semilogy(f, p)
        ax2.set_xlabel("Frecuencia (Hz)")
        ax2.set_ylabel("PSD")
        st.pyplot(fig2)
        
    # Tab 3 y 4: Cálculo de fatiga
    win_sec = 0.5
    w_len = int(win_sec * fs)
    segs = [sig_f[i:i+w_len] for i in range(0, len(sig_f)-w_len, w_len)]
    metrics = [get_metrics(seg, fs) for seg in segs]
    mnf_v, mdf_v, rms_v, _, _ = zip(*metrics)
    t_segs = np.arange(len(segs)) * win_sec

    with tab3:
        st.subheader("Evolución de Frecuencia y Amplitud")
        fig3, ax3 = plt.subplots(2, 1, figsize=(10, 6))
        ax3[0].plot(t_segs, mnf_v, color='blue', label="MNF")
        ax3[0].plot(t_segs, mdf_v, color='red', label="MDF")
        ax3[0].legend()
        ax3[1].plot(t_segs, rms_v, color='orange', label="RMS")
        ax3[1].legend()
        st.pyplot(fig3)

    with tab4:
        st.subheader("Estado de Fatiga")
        drop = ((mnf_v[-1] - mnf_v[0]) / mnf_v[0]) * 100
        st.metric("Caída MNF (%)", f"{drop:.2f}%")
        if drop < -15:
            st.error("Nivel: FATIGA DETECTADA")
        else:
            st.success("Nivel: MUSCULO FRESCO")

else:
    st.info("Por favor, sube un archivo CSV para empezar.")
