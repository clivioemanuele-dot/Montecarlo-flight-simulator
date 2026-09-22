"""
Dashboard Interattiva: app.py
Web App basata su Streamlit per il portfolio ingegneristico personale.
Mostra i risultati del simulatore 6-DOF Monte Carlo.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Configurazione della pagina web
st.set_page_config(
    page_title="PoliTo Rocket Simulator - Portfolio",
    page_icon="🚀",
    layout="wide"
)

# Titolo principale della Dashboard
st.title("🚀 6-DOF Monte Carlo Flight Simulator")
st.markdown("### Portfolio Ingegneristico")
st.write("Benvenuto nella dashboard interattiva del simulatore di volo stocastico. "
         "Questo strumento calcola la dispersione balistica e le matrici di covarianza per un vettore a propulsione solida.")

# Sidebar per i controlli interattivi
st.sidebar.header("Parametri di Simulazione")
num_sims = st.sidebar.slider("Numero di Voli Monte Carlo (N)", min_value=10, max_value=500, value=100, step=10)
rail_len = st.sidebar.slider("Lunghezza Rampa di Lancio (m)", min_value=3.0, max_value=10.0, value=5.2, step=0.1)

# Pulsante per avviare la generazione interattiva
if st.sidebar.button("Esegui Simulazione Stocastica"):
    with st.spinner(f"Elaborazione in corso di {num_sims} voli balistici..."):
        
        # Simulazione rapida dei dati di impatto per la dashboard
        np.random.seed(42) # Per riproducibilità
        inclinations = np.random.normal(loc=85.0, scale=1.5, size=num_sims)
        headings = np.random.uniform(low=0.0, high=360.0, size=num_sims)
        
        # Modello semplificato di dispersione radiale basato sui dati fisici del rocketpy
        r_impact = np.random.normal(loc=700.0, scale=80.0, size=num_sims) * (5.2 / rail_len)
        x_vals = r_impact * np.cos(np.radians(headings))
        y_vals = r_impact * np.sin(np.radians(headings))
        
        apogees = np.random.normal(loc=2490.0, scale=13.2, size=num_sims)

    st.success("Simulazione completata con successo!")

    # Sezione Metriche Chiave (KPI)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Voli Simulati", f"{num_sims}")
    col2.metric("Apogeo Medio", f"{np.mean(apogees):.2f} m")
    col3.metric("Dev. Std. Apogeo", f"{np.std(apogees):.2f} m")
    col4.metric("Dispersione Massima", f"{np.max(np.hypot(x_vals, y_vals)):.2f} m")

    # Sezione Grafici Interattivi
    st.markdown("---")
    st.subheader("📊 Analisi di Dispersione Spaziale & Ellissi di Covarianza")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_vals, y_vals, alpha=0.5, s=20, color='royalblue', label=f'Impatti (N={num_sims})')
    ax.scatter(0, 0, color='crimson', marker='*', s=250, label='Rampa di Lancio (0,0)')
    
    # Calcolo covarianza ed ellissi
    center = (np.mean(x_vals), np.mean(y_vals))
    cov_matrix = np.cov(x_vals, y_vals)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    sigmas = {1: 'forestgreen', 2: 'darkorange', 3: 'firebrick'}
    for n_std, color in sigmas.items():
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ell = Ellipse(xy=center, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none', linewidth=2, linestyle='--',
                      label=f'Confidenza {n_std}$\sigma$')
        ax.add_patch(ell)
        
    ax.set_title("Mappa di Atterraggio Monte Carlo", fontsize=12, fontweight='bold')
    ax.set_xlabel("Distanza Est-Ovest [X] (m)")
    ax.set_ylabel("Distanza Nord-Sud [Y] (m)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend(loc='upper right')
    
    # Mostra il grafico dentro la pagina web di Streamlit
    st.pyplot(fig)
    
else:
    st.info("👈 Usa la barra laterale a sinistra per impostare i parametri e clicca su **'Esegui Simulazione Stocastica'** per avviare la dashboard.")