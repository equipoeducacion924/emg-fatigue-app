import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Análisis EMG", layout="wide")

st.title("⚡ Aplicación de Fatiga EMG")
st.sidebar.header("Configuración")

# Subir archivo
archivo = st.file_uploader("Sube tu archivo de señal EMG (CSV)", type=["csv"])

if archivo is not None:
    # Leer datos
    df = pd.read_csv(archivo)
    
    st.success("Archivo cargado correctamente")
    
    # Mostrar datos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Estadísticas básicas")
        st.write(df.describe())

    # Gráfica de la señal
    st.subheader("Visualización de la Señal EMG")
    columna = st.selectbox("Selecciona la columna de la señal", df.columns)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[columna], color='#00FFAA')
    ax.set_title(f"Señal: {columna}")
    ax.set_xlabel("Muestras")
    ax.set_ylabel("Amplitud (mV)")
    st.pyplot(fig)

else:
    st.info("💡 Por favor, sube un archivo CSV para comenzar el análisis.")
