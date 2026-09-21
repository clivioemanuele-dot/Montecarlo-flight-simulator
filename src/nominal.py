"""
Modulo: nominal.py
Definizione della baseline nominale per il simulatore Monte Carlo a 6-DOF.
Scritto in Python Object-Oriented con Type Hinting per standard aerospaziali.
"""

from rocketpy import Environment, SolidMotor, Rocket, Flight
import numpy as np

# --- INIZIO PATCH PER COMPATIBILITÀ NUMPY 2.x ---
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
# --- FINE PATCH ---

class NominalSimulation:
    """
    Classe base che incapsula il modello di volo nominale (senza perturbazioni).
    Verrà utilizzata come 'Ground Truth' per generare i voli dispersi.
    """
    
    def __init__(self) -> None:
        self.env = self._create_environment()
        self.motor = self._create_motor()
        self.rocket = self._create_rocket(self.motor)
        
    def _create_environment(self) -> Environment:
        """Inizializza l'atmosfera standard e le coordinate di lancio."""
        env = Environment(latitude=45.06, longitude=7.66, elevation=250)
        env.set_date((2026, 10, 25, 12, 0, 0))
        env.set_atmospheric_model(type="standard_atmosphere")
        return env

    def _create_motor(self) -> SolidMotor:
        """Definisce il propulsore a propellente solido con curva di spinta."""
        thrust_curve = [[0.0, 0.0], [0.1, 2000.0], [2.0, 1500.0], [3.9, 0.0]]
        
        motor = SolidMotor(
            thrust_source=thrust_curve,
            dry_mass=1.815,
            dry_inertia=(0.125, 0.125, 0.002),
            nozzle_radius=0.033,
            grain_number=5,
            grain_density=1815,
            grain_outer_radius=0.033,
            grain_initial_inner_radius=0.015,
            grain_initial_height=0.12,
            grain_separation=0.005,
            nozzle_position=0,
            burn_time=3.9,
            coordinate_system_orientation="nozzle_to_combustion_chamber",
            grains_center_of_mass_position=-0.06, 
            center_of_dry_mass_position=-0.06    
        )
        return motor

    def _create_rocket(self, motor: SolidMotor) -> Rocket:
        """Assembla il vettore spaziale (Massa, Aerodinamica, Motore, Paracadute)."""
        rocket = Rocket(
            radius=0.0635,
            mass=14.426,
            inertia=(6.321, 6.321, 0.034),
            power_off_drag=0.45,
            power_on_drag=0.45,
            center_of_mass_without_motor=0,
            coordinate_system_orientation="tail_to_nose"
        )
        
        rocket.add_motor(motor, position=-1.255)
        
        rocket.add_trapezoidal_fins(
            n=4, root_chord=0.120, tip_chord=0.040, span=0.100,
            position=-1.0495, sweep_length=0.068
        )
        
        rocket.add_parachute(
            name="Main_Chute",
            cd_s=10.0,
            trigger=800,
            sampling_rate=105,
            lag=1.5,
            noise=(0, 8.3, 0.5)
        )
        
        return rocket

    def run_flight(self) -> Flight:
        """Risolve le equazioni del moto a 6 gradi di libertà."""
        print("Avvio simulazione nominale 6-DOF...")
        flight = Flight(
            rocket=self.rocket,
            environment=self.env,
            rail_length=5.2,
            inclination=85,
            heading=0 
        )
        return flight

if __name__ == "__main__":
    sim = NominalSimulation()
    nominal_flight = sim.run_flight()
    
    print(f"\n--- RISULTATI VOLO NOMINALE ---")
    print(f"Apogeo raggiunto : {nominal_flight.apogee:.2f} m")
    print(f"Tempo di volo    : {nominal_flight.t_final:.2f} s")
    print(f"Velocità massima : {nominal_flight.max_speed:.2f} m/s")
    print(f"Coordinata X impatto : {nominal_flight.x_impact:.2f} m")
    print(f"Coordinata Y impatto : {nominal_flight.y_impact:.2f} m")