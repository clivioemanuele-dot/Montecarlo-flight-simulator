"""
Modulo: monte_carlo.py
Simulatore Monte Carlo avanzato a 6-DOF con esportazione dati CSV
ed elaborazione statistica per il PoliTo Rocket Team.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
from rocketpy import Flight
from nominal import NominalSimulation

class AdvancedMonteCarlo(NominalSimulation):
    def __init__(self, num_simulations: int = 1000):
        super().__init__()
        self.num_simulations = num_simulations
        self.results_data = []
        
    def run_simulation(self):
        print(f"\n[AVVIO] Simulazione Monte Carlo su larga scala: {self.num_simulations} voli in corso...")
        
        for _ in tqdm(range(self.num_simulations), desc="Elaborazione traiettorie"):
            # Perturazioni stocastiche realistiche
            stochastic_inclination = np.random.normal(loc=85.0, scale=1.5)
            stochastic_heading = np.random.uniform(low=0.0, high=360.0)
            
            flight = Flight(
                rocket=self.rocket,
                environment=self.env,
                rail_length=5.2,
                inclination=stochastic_inclination,
                heading=stochastic_heading
            )
            
            # Raccolta metriche chiave per il database
            self.results_data.append({
                "apogee": flight.apogee,
                "t_final": flight.t_final,
                "max_speed": flight.max_speed,
                "x_impact": flight.x_impact,
                "y_impact": flight.y_impact,
                "inclination": stochastic_inclination,
                "heading": stochastic_heading
            })
            
        self._export_to_csv()
        self._generate_statistical_report()
        self._plot_advanced_results()

    def _export_to_csv(self):
        """Esporta i dati grezzi in un file CSV standard industriale."""
        filename = "monte_carlo_dataset.csv"
        keys = self.results_data[0].keys()
        
        with open(filename, "w", newline="", encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.results_data)
            
        print(f"[DATI] Dataset telemetrico salvato con successo in '{filename}'")

    def _generate_statistical_report(self):
        """Calcola le metriche di dispersione per la commissione tecnica."""
        apogees = [r["apogee"] for r in self.results_data]
        x_imp = [r["x_impact"] for r in self.results_data]
        y_imp = [r["y_impact"] for r in self.results_data]
        
        print("\n" + "="*40)
        print("   REPORT STATISTICO MONTE CARLO (N=1000)")
        print("="*40)
        print(f"Apogeo Medio       : {np.mean(apogees):.2f} m")
        print(f"Dev. Std. Apogeo   : {np.std(apogees):.2f} m")
        print(f"Apogeo Min / Max   : {np.min(apogees):.2f} m / {np.max(apogees):.2f} m")
        print(f"Distanza Media X   : {np.mean(x_imp):.2f} m")
        print(f"Distanza Media Y   : {np.mean(y_imp):.2f} m")
        print("="*40 + "\n")

    def _plot_advanced_results(self):
        """Genera il grafico di dispersione ad alta densità con ellissi."""
        x_vals = np.array([r["x_impact"] for r in self.results_data])
        y_vals = np.array([r["y_impact"] for r in self.results_data])
        
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        ax.scatter(x_vals, y_vals, alpha=0.3, s=12, color='royalblue', label='Impatti (N=1000)')
        ax.scatter(0, 0, color='crimson', marker='*', s=300, label='Rampa di Lancio (0,0)')
        
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
            
        plt.title("Analisi di Dispersione Statistica 6-DOF (Monte Carlo 1000 Voli)", fontsize=13, fontweight='bold')
        plt.xlabel("Distanza Est-Ovest [X] (m)", fontsize=11)
        plt.ylabel("Distanza Nord-Sud [Y] (m)", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axhline(0, color='black', linewidth=0.6)
        plt.axvline(0, color='black', linewidth=0.6)
        plt.legend(loc='upper right')
        
        output_image = "dispersione_1000_voli.png"
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"[GRAFICO] Visualizzazione avanzata salvata in '{output_image}'")

if __name__ == "__main__":
    sim_mc = AdvancedMonteCarlo(num_simulations=1000)
    sim_mc.run_simulation()