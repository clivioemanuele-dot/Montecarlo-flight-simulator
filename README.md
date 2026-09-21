# Montecarlo-flight-simulator

# 🚀 6-DOF Monte Carlo Flight Simulator

> Sviluppato un simulatore balistico e stocastico avanzato a 6 Gradi di Libertà (6-DOF) scritto in Python, focalizzato sull'ingegneria aerospaziale, programmazione ad oggetti (OOP) e analisi statistica della dispersione.

---

## 📋 Panoramica del Progetto
Questo software è stato progettato per simulare la traiettoria di volo di un razzo a propulsione solida e prevedere la sua zona di atterraggio stimata in condizioni operative reali. Attraverso l'implementazione del **Metodo Monte Carlo** (N=1000 voli stocastici), il simulatore tiene conto delle incertezze di lancio (tolleranze di rampa e variabilità azimutale) calcolando le matrici di covarianza e le relative **ellissi di confidenza (1\sigma, 2\sigma, 3\sigma)**.

---

## 🛠️ Stack Tecnologico
* **Linguaggio:** Python 3.13
* **Aerospazio & Fisica:** [RocketPy](https://github.com/RocketPyTeam/RocketPy) (Simulazione 6-DOF del volo)
* **Calcolo Numerico:** NumPy, SciPy (Algebra lineare e matrici di covarianza)
* **Visualizzazione:** Matplotlib (Generazione di grafici statistici ad alta risoluzione)
* **Ingegneria del Software:** Programmazione a Oggetti (OOP), Type Hinting, Gestione Dipendenze, Export CSV

---

## 📂 Struttura del Progetto
```text
polito_mc_sim/
│
├── src/
│   ├── nominal.py          # Modello di base (Ground Truth - Volo Nominale)
│   └── monte_carlo.py      # Motore stocastico, analisi statistica e export CSV
│
├── venv/                   # Ambiente virtuale isolato
├── requirements.txt        # Distinta base delle dipendenze
├── monte_carlo_dataset.csv # Dataset telemetrico dei 1000 voli
└── dispersione_1000_voli.png # Grafico di dispersione con ellissi di confidenza
