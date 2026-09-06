"""
environment.py — Modello della griglia di volo del drone.
Definisce i tipi di cella (CellType), la struttura dati di ogni cella (GridCell)
e la griglia completa (Grid) con le operazioni di accesso, costo e aggiornamento
dello stato delle consegne.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np

# Tipi di cella

class CellType(IntEnum):
    """
    Tipi possibili per una cella della griglia.
    Usa IntEnum per compatibilità con numpy e confronti efficienti.
    """
    EMPTY    = 0   # Cella libera — attraversabile a costo base 1.0
    START    = 1   # Posizione di partenza e ritorno del drone (base)
    BLOCKED  = 3   # No-fly zone — ostacolo invalicabile (costo infinito)
    DELIVERY = 4   # Punto di consegna — obiettivo parziale da raggiungere
    WIND     = 5   # Zona di vento avverso — costo di attraversamento raddoppiato


# Costo base per ogni tipo di cella (prima del moltiplicatore vento)
BASE_COST: dict[CellType, float] = {
    CellType.EMPTY:    1.0,
    CellType.START:    1.0,   # Tornare alla base ha costo normale
    CellType.BLOCKED:  float("inf"), # Costo infinito per celle bloccate
    CellType.DELIVERY: 1.0,
    CellType.WIND:     1.0,   # Il moltiplicatore viene applicato in step_cost()
}

# Moltiplicatore di costo applicato alle celle WIND
WIND_MULTIPLIER = 2.0


# Cella della griglia

@dataclass
class GridCell:
    cell_type: CellType # Tipo della cella (EMPTY, START, BLOCKED, DELIVERY, WIND)
    delivered: bool = False # Solo per DELIVERY: True se la consegna è stata effettuata

    def is_passable(self) -> bool:
        """Restituisce True se il drone può entrare nella cella (non è BLOCKED)."""
        return self.cell_type != CellType.BLOCKED

    def has_pending_delivery(self) -> bool:
        """Restituisce True se la cella è un punto di consegna non ancora servito."""
        return self.cell_type == CellType.DELIVERY and not self.delivered

    def __repr__(self) -> str: # Rappresentazione testuale per debug
        return f"GridCell({self.cell_type.name})"



# Griglia di volo

class Grid:

    def __init__(self, cells: np.ndarray) -> None:
        if cells.ndim != 2:
            raise ValueError("L'array celle deve essere 2D (righe × colonne).")
        self.rows, self.cols = cells.shape
        self.cells: np.ndarray = cells

        # Individua le posizioni speciali alla creazione della griglia
        self.start_pos          = self._find_unique(CellType.START, required=True)
        self.delivery_positions = self._find_all(CellType.DELIVERY)

    def __getitem__(self, pos: tuple[int, int]) -> GridCell:
        """Accesso diretto a una cella tramite (riga, colonna)."""
        return self.cells[pos[0], pos[1]]

    def in_bounds(self, row: int, col: int) -> bool:
        """Restituisce True se le coordinate (row, col) sono dentro i limiti della griglia."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_passable(self, row: int, col: int) -> bool:
        """
        Restituisce True se la cella è dentro la griglia e non è BLOCKED.
        Combina in_bounds() e GridCell.is_passable().
        """
        return self.in_bounds(row, col) and self.cells[row, col].is_passable()

    def step_cost(self, row: int, col: int) -> float:
        """
        Calcola il costo di attraversamento della cella (row, col).

        Formula: BASE_COST[cell_type] × wind_multiplier
        - wind_multiplier = 2.0 per celle WIND, 1.0 altrimenti.
        - Restituisce float('inf') per celle BLOCKED.
        """
        cell = self.cells[row, col]
        if not cell.is_passable():
            return float("inf")
        wind_mult = WIND_MULTIPLIER if cell.cell_type == CellType.WIND else 1.0
        return BASE_COST[cell.cell_type] * wind_mult

    def mark_delivered(self, row: int, col: int) -> bool:
        """
        Segna la consegna nella cella (row, col) come completata.
        Restituisce True se la consegna era pendente (e quindi è stata effettuata),
        False se la cella non era una DELIVERY pendente.
        """
        cell = self.cells[row, col]
        if cell.has_pending_delivery():
            cell.delivered = True
            return True
        return False

    def reset_deliveries(self) -> None:
        """
        Reimposta tutte le consegne come non effettuate (delivered=False).
        Usato per garantire lo stesso stato iniziale tra run di algoritmi diversi.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r, c].delivered = False

    def pending_deliveries(self) -> list[tuple[int, int]]:
        """Restituisce le posizioni delle consegne ancora da effettuare."""
        return [p for p in self.delivery_positions
                if self.cells[p[0], p[1]].has_pending_delivery()]

    def manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """
        Calcola la distanza di Manhattan tra due celle a=(r1,c1) e b=(r2,c2).
        d = |r1-r2| + |c1-c2|

        Usata come euristica ammissibile in A*, IDA*, RBFS e SMA*:
        è un lower bound del costo reale poiché ogni passo costa almeno 1.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _find_unique(self, cell_type: CellType, required: bool = True) -> Optional[tuple[int, int]]:
        """
        Trova l'unica cella di un determinato tipo.
        Lancia ValueError se ne esistono più di una o se required=True e non ne esiste nessuna.
        """
        positions = self._find_all(cell_type)
        if not positions:
            if required:
                raise ValueError(f"Nessuna cella {cell_type.name} trovata nella griglia.")
            return None
        if len(positions) > 1:
            raise ValueError(
                f"Trovate {len(positions)} celle {cell_type.name}: deve essercene esattamente una."
            )
        return positions[0]

    def _find_all(self, *cell_types: CellType) -> list[tuple[int, int]]:
        """Restituisce le coordinate di tutte le celle che corrispondono ai tipi indicati."""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.cells[r, c].cell_type in cell_types
        ]

    # Mappa simbolo ASCII per ogni tipo di cella (usata da ascii_render)
    CELL_SYMBOLS = {
        CellType.EMPTY:    ".",
        CellType.START:    "S",
        CellType.BLOCKED:  "X",
        CellType.DELIVERY: "D",
        CellType.WIND:     "W",
    }

    def ascii_render(self, drone_pos: Optional[tuple[int, int]] = None) -> str:
        """
        Restituisce una rappresentazione ASCII della griglia.
        - '@' indica la posizione corrente del drone (se fornita).
        - Lettere minuscole indicano consegne già effettuate.
        """
        lines = []
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if drone_pos and (r, c) == drone_pos:
                    row_str += "@"
                else:
                    cell = self.cells[r, c]
                    sym  = self.CELL_SYMBOLS[cell.cell_type]
                    # Lettera minuscola = consegna completata
                    row_str += sym.lower() if cell.delivered else sym
            lines.append(row_str)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Grid({self.rows}x{self.cols}, start={self.start_pos})"
