"""
sensors.py — Modulo sensori del drone.

Simula la percezione del drone sull'ambiente: tiene traccia della posizione
corrente, del numero di passi effettuati e rileva le condizioni di vento
nella cella occupata.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Grid


# Snapshot dello stato sensori
@dataclass
class SensorState:
    """
    Snapshot immutabile dello stato dei sensori.
    Usato per salvare e ripristinare la posizione del drone
    tra un run di algoritmo e l'altro.
    """
    position:    tuple[int, int]  # Posizione corrente (riga, colonna)
    steps_taken: int = 0          # Numero totale di passi effettuati

    def clone(self) -> "SensorState":
        """Restituisce una copia indipendente dello snapshot."""
        return SensorState(position=self.position, steps_taken=self.steps_taken)


# Classe principale sensori
class DroneSensors:
    """
    Gestisce i sensori del drone: posizione e rilevamento del vento.

    Viene aggiornato ad ogni passo del drone tramite update().
    Il metodo get_state() / set_state() consente di salvare e ripristinare
    la configurazione — necessario per confrontare più algoritmi sullo stesso
    stato iniziale senza reinizializzare l'intero sistema.
    """

    def __init__(self, grid: "Grid", start_pos: tuple[int, int]) -> None:
        self._grid       = grid
        self.position    = start_pos   # Posizione iniziale del drone
        self.steps_taken = 0           # Contatore passi (incrementato ad ogni update)

   
    # Rilevamento vento
    def wind_at(self, row: int, col: int) -> bool:
        """
        Rileva se la cella (row, col) è una zona di vento avverso (CellType.WIND).
        Il vento raddoppia il costo di attraversamento (WIND_MULTIPLIER = 2.0).
        """
        from environment import CellType
        return self._grid[row, col].cell_type == CellType.WIND

    def wind_multiplier_at(self, row: int, col: int) -> float:
        """
        Restituisce il moltiplicatore di costo per la cella (row, col):
        - 2.0 se la cella è WIND
        - 1.0 altrimenti
        """
        from environment import WIND_MULTIPLIER
        return WIND_MULTIPLIER if self.wind_at(row, col) else 1.0

    # Aggiornamento stato
    def update(self, new_pos: tuple[int, int]) -> dict:
        """
        Aggiorna i sensori dopo uno spostamento del drone.
        Incrementa il contatore passi, aggiorna la posizione e
        restituisce un dizionario con i dati sensoriali correnti.

        Restituisce:
            {
                "position":    (row, col),  # nuova posizione
                "steps_taken": int,         # passi totali
                "wind":        bool,        # True se la nuova cella è WIND
            }
        """
        self.position = new_pos
        self.steps_taken += 1
        r, c = new_pos
        return {
            "position":    new_pos,
            "steps_taken": self.steps_taken,
            "wind":        self.wind_at(r, c),
        }

    def reset(self, start_pos: tuple[int, int]) -> None:
        """
        Reimposta i sensori alla posizione iniziale con passi=0.
        Usato da compare_all() in controller.py per garantire che ogni
        algoritmo parta dallo stesso stato iniziale.
        """
        self.position    = start_pos
        self.steps_taken = 0

    # Snapshot
    def get_state(self) -> SensorState:
        """Salva lo stato corrente dei sensori come snapshot."""
        return SensorState(position=self.position, steps_taken=self.steps_taken)

    def set_state(self, state: SensorState) -> None:
        """Ripristina uno stato precedentemente salvato con get_state()."""
        self.position    = state.position
        self.steps_taken = state.steps_taken

    # Utility
    def status(self) -> str:
        """Restituisce una stringa di riepilogo dello stato corrente."""
        return f"[Sensors] pos={self.position} | steps={self.steps_taken}"

    def __repr__(self) -> str:
        return self.status()
