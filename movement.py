"""
movement.py — Regole di spostamento del drone nella griglia.

Definisce le direzioni cardinali (Direction), il motore di movimento
(MovementEngine) e il risultato di ogni mossa (MoveResult).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Grid
    from sensors import DroneSensors


# Direzioni cardinali

class Direction(Enum):
    """
    Rappresenta le quattro direzioni di spostamento del drone.
    Il valore è il delta (driga, dcolonna) applicato alla posizione corrente.
    """
    NORTH = (-1,  0)   # Riga diminuisce (verso l'alto nella griglia)
    SOUTH = ( 1,  0)   # Riga aumenta    (verso il basso)
    EAST  = ( 0,  1)   # Colonna aumenta (verso destra)
    WEST  = ( 0, -1)   # Colonna diminuisce (verso sinistra)

    def apply(self, pos: tuple[int, int]) -> tuple[int, int]:
        """
        Applica la direzione a una posizione e restituisce la nuova posizione.
        Esempio: Direction.EAST.apply((2, 3)) → (2, 4)
        """
        return (pos[0] + self.value[0], pos[1] + self.value[1])

    def __str__(self) -> str:
        """Rappresentazione simbolica della direzione (usata nel log)."""
        return {
            Direction.NORTH: "^",
            Direction.SOUTH: "v",
            Direction.EAST:  ">",
            Direction.WEST:  "<",
        }[self]


# Lista di tutte le direzioni — usata in valid_moves() per iterare
ALL_DIRECTIONS = list(Direction)


# Risultato di una mossa

@dataclass
class MoveResult:
    """
    Risultato restituito da MovementEngine.execute_move().

    Contiene tutte le informazioni sull'esito della mossa:
    posizione precedente/nuova, eventuali consegne effettuate,
    rilevamento vento e motivo del fallimento.
    """
    success:       bool
    old_pos:       tuple[int, int]   # Posizione prima della mossa
    new_pos:       tuple[int, int]   # Posizione dopo la mossa (o tentata)
    direction:     Direction
    steps_taken:   int  = 0          # Totale passi effettuati dal drone
    delivered:     bool = False      # True se è avvenuta una consegna
    delivery_type: Optional[str] = None  # Nome del tipo di cella consegnata
    is_wind:       bool = False      # True se la nuova cella è WIND
    is_start:      bool = False      # True se il drone è tornato alla base
    fail_reason:   str  = ""         # Motivo del fallimento (se success=False)

    def __str__(self) -> str:
        if not self.success:
            return f"Mossa FALLITA ({self.fail_reason})"
        tag = ""
        if self.delivered: tag += f" [CONSEGNA {self.delivery_type}]"
        if self.is_wind:   tag += " [VENTO]"
        return f"{self.direction} {self.old_pos}->{self.new_pos} | step={self.steps_taken}{tag}"


# Motore di movimento

class MovementEngine:
    """
    Implementa le regole di spostamento del drone nella griglia.

    Verifica la validità delle mosse (bounds + passabilità), aggiorna
    i sensori e gestisce le consegne automatiche nelle celle DELIVERY.
    """

    def __init__(self, grid: "Grid", sensors: "DroneSensors") -> None:
        self._grid    = grid
        self._sensors = sensors

    def valid_moves(
        self, pos: Optional[tuple[int, int]] = None
    ) -> list[tuple[Direction, tuple[int, int]]]:
        """
        Restituisce la lista delle mosse valide dalla posizione indicata.
        Se pos=None usa la posizione corrente dei sensori.

        Restituisce: lista di (direzione, nuova_posizione) per ogni mossa valida.
        Una mossa è valida se la destinazione è in_bounds e is_passable.
        """
        if pos is None:
            pos = self._sensors.position
        result = []
        for direction in ALL_DIRECTIONS:
            new_pos = direction.apply(pos)
            r, c = new_pos
            if self._grid.in_bounds(r, c) and self._grid.is_passable(r, c):
                result.append((direction, new_pos))
        return result

    def valid_positions(self, pos: Optional[tuple[int, int]] = None) -> list[tuple[int, int]]:
        """Restituisce solo le posizioni (senza direzione) delle mosse valide."""
        return [p for _, p in self.valid_moves(pos)]

    def execute_move(
        self,
        direction: Direction,
        pos: Optional[tuple[int, int]] = None,
    ) -> MoveResult:
        """
        Esegue un singolo passo del drone nella direzione indicata.

        Processo:
        1. Verifica che la destinazione sia in bounds e passabile.
        2. Aggiorna i sensori (posizione + contatore passi).
        3. Se la cella di destinazione è una DELIVERY pendente, effettua la consegna.
        4. Restituisce un MoveResult con l'esito completo della mossa.

        Se pos=None usa la posizione corrente dei sensori.
        """
        from environment import CellType
        if pos is None:
            pos = self._sensors.position
        old_pos = pos
        new_pos = direction.apply(pos)
        r, c = new_pos

        # Verifica bounds
        if not self._grid.in_bounds(r, c):
            return MoveResult(False, old_pos, new_pos, direction,
                              fail_reason="fuori dai limiti della griglia")

        # Verifica passabilità (no-fly zone)
        if not self._grid.is_passable(r, c):
            return MoveResult(False, old_pos, new_pos, direction,
                              fail_reason="cella non attraversabile (X)")

        # Aggiorna sensori
        sensor_data   = self._sensors.update(new_pos)
        cell          = self._grid[r, c]
        delivered     = False
        delivery_type = None

        # Consegna automatica: se il drone entra in una cella DELIVERY pendente
        if cell.has_pending_delivery():
            delivered = self._grid.mark_delivered(r, c)
            if delivered:
                delivery_type = cell.cell_type.name

        return MoveResult(
            success       = True,
            old_pos       = old_pos,
            new_pos       = new_pos,
            direction     = direction,
            steps_taken   = sensor_data["steps_taken"],
            delivered     = delivered,
            delivery_type = delivery_type,
            is_wind       = sensor_data["wind"],
            is_start      = cell.cell_type == CellType.START,
        )

    def execute_path(
        self, path: list[tuple[int, int]], verbose: bool = False
    ) -> list[MoveResult]:
        """
        Esegue una sequenza di posizioni come percorso.
        Determina automaticamente la direzione tra posizioni consecutive
        e chiama execute_move() per ogni passo.

        Se verbose=True stampa ogni MoveResult.
        Si interrompe al primo passo fallito.
        """
        results = []
        for i in range(1, len(path)):
            prev, curr = path[i - 1], path[i]
            dr, dc = curr[0] - prev[0], curr[1] - prev[1]
            # Ricava la direzione dal delta (dr, dc)
            direction = next(
                (d for d in ALL_DIRECTIONS if d.value == (dr, dc)), None
            )
            if direction is None:
                raise ValueError(
                    f"Passo non valido nel percorso: {prev} -> {curr} "
                    f"(le celle non sono adiacenti)"
                )
            result = self.execute_move(direction, pos=prev)
            results.append(result)
            if verbose:
                print(result)
            if not result.success:
                break
        return results

    def path_cost(self, path: list[tuple[int, int]]) -> float:
        """
        Calcola il costo totale di un percorso sommando i costi di attraversamento
        di ogni cella visitata (esclusa la prima, che è la posizione di partenza).
        """
        return sum(self._grid.step_cost(r, c) for r, c in path[1:])

    def status(self) -> str:
        """Stringa di riepilogo: posizione corrente e mosse valide disponibili."""
        moves = self.valid_moves()
        return (
            f"[Movement] pos={self._sensors.position} | "
            f"mosse valide: {len(moves)} "
            f"({', '.join(str(d) for d, _ in moves)})"
        )
