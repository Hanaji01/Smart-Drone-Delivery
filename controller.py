"""
controller.py — Algoritmi di ricerca e pianificazione del percorso.

Implementa nove algoritmi di ricerca per pianificare il percorso ottimale
del drone: BFS, DFS, IDS, UCS, Greedy, A*, IDA*, RBFS, SMA*.

L'algoritmo principale è A*: ottimale, informato e adatto a grafi con
costi eterogenei come questo dominio (celle WIND a costo doppio).

"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from environment import Grid
from sensors import DroneSensors


# Configurazione: algoritmi abilitati per il confronto (tasto C)
ENABLED_ALGORITHMS: dict[str, bool] = {
    "BFS":    False,
    "DFS":    False,
    "IDS":    False,
    "UCS":    False,
    "Greedy": False,
    "A*":     True,   
    "IDA*":   False,
    "RBFS":   False,
    "SMA*":   False,
}

# Strutture dati per la ricerca
@dataclass(frozen=True)
class SearchState:
    """
    Stato del problema di ricerca.

    Rappresenta la configurazione completa del sistema in un dato momento:
    - position: coordinate (riga, colonna) del drone.
    - pending_deliveries: frozenset delle posizioni delle consegne non ancora
      effettuate. L'uso di frozenset garantisce che due stati con le stesse
      consegne pendenti (indipendentemente dall'ordine) siano identici e
      hashable — necessario per inserirli in set e dizionari.
    - steps: numero di passi effettuati; usato come tie-breaker.

    Stato goal: pending_deliveries=∅ e position=base.
    """
    position:           tuple[int, int]
    pending_deliveries: frozenset        # Posizioni consegne ancora da effettuare
    steps:              int = field(compare=False, hash=False) # Tie-breaker per priorità uguale

    def __lt__(self, other: "SearchState") -> bool:
        # Necessario per ordinamento in heapq quando f_cost è uguale
        return self.steps < other.steps


@dataclass
class SearchNode:
    """
    Nodo dell'albero di ricerca.

    Contiene lo stato, il riferimento al padre (per ricostruire il percorso),
    l'azione che ha portato a questo nodo, il costo g(n) dal nodo iniziale
    e l'euristica h(n).

    La proprietà f_cost = g(n) + h(n) è usata da A*, IDA*, RBFS e SMA*
    per ordinare la frontiera.
    """
    state:  SearchState
    parent: Optional["SearchNode"]
    action: Optional[tuple[int, int]]  # Posizione raggiunta con questa mossa
    g_cost: float                      # g(n): costo cumulativo dal nodo iniziale
    h_cost: float = 0.0                # h(n): stima euristica al goal
    depth:  int   = 0                  # Profondità nell'albero di ricerca

    @property
    def f_cost(self) -> float:
        """f(n) = g(n) + h(n): stima del costo totale del percorso passante per n."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: "SearchNode") -> bool:
        # Necessario per heapq (priority queue)
        return self.f_cost < other.f_cost

    def path(self) -> list[tuple[int, int]]:
        """
        Ricostruisce il percorso dal nodo iniziale a questo nodo
        risalendo la catena di nodi padre. Restituisce solo le azioni
        (posizioni raggiunte), senza il nodo iniziale.
        """
        nodes, node = [], self
        while node is not None:
            if node.action is not None:
                nodes.append(node.action)
            node = node.parent
        nodes.reverse()
        return nodes

    def full_path(self, start: tuple[int, int]) -> list[tuple[int, int]]:
        """Restituisce il percorso completo inclusa la posizione di partenza."""
        return [start] + self.path()


@dataclass
class SearchResult:
    """
    Risultato di una ricerca. Raccoglie tutte le metriche di prestazione.

    Attributi:
        algorithm:      nome dell'algoritmo usato.
        path:           sequenza di celle del percorso ottimale (inclusa la base).
        total_cost:     costo energetico totale del percorso.
        nodes_expanded: nodi espansi — misura del lavoro computazionale.
        max_frontier:   dimensione massima della frontiera — misura della memoria.
        time_seconds:   tempo di esecuzione in secondi.
        success:        True se è stato trovato un percorso valido.
        message:        messaggio di errore o informativo (se success=False).
    """
    algorithm:      str
    path:           list[tuple[int, int]]
    total_cost:     float
    nodes_expanded: int
    max_frontier:   int
    time_seconds:   float
    success:        bool
    message:        str = ""

    def __str__(self) -> str:
        if not self.success:
            return f"[{self.algorithm}] FALLITO: {self.message}"
        return (
            f"[{self.algorithm}] "
            f"passi={len(self.path)-1} | costo={self.total_cost:.2f} | "
            f"nodi={self.nodes_expanded} | tempo={self.time_seconds*1000:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Controller — implementa tutti gli algoritmi di ricerca
# ---------------------------------------------------------------------------

class SearchController:
    """
    Pianificatore del percorso del drone.

    Implementa nove algoritmi di ricerca su uno spazio degli stati composito:
    ogni stato include posizione del drone + insieme delle consegne pendenti.

    Uso tipico:
        controller = SearchController(grid, sensors)
        result = controller.astar()
        if result.success:
            path = result.path  # lista di (row, col)
    """

    def __init__(self, grid: Grid, sensors: DroneSensors) -> None:
        self._grid    = grid
        self._sensors = sensors

    # ------------------------------------------------------------------
    # Euristica
    # ------------------------------------------------------------------

    def _heuristic(
        self, pos: tuple[int, int], pending: frozenset, weight: float = 1.0
    ) -> float:
        """
        Euristica ammissibile per il problema multi-consegna.

        Stima il costo rimanente come lower bound:
        - Se ci sono consegne pendenti:
            h(n) = manhattan(pos, consegna_più_vicina)
                 + manhattan(consegna_più_lontana, base)
        - Se non ci sono consegne pendenti:
            h(n) = manhattan(pos, base)

        L'euristica è ammissibile perché:
        1. La distanza di Manhattan non tiene conto degli ostacoli.
        2. Non considera il costo delle celle WIND (ogni passo costa ≥ 1).
        Quindi non sovrastima mai il costo reale → A* è ottimale.

        Il parametro weight consente A* ponderato (weight > 1.0):
        più veloce ma non garantisce ottimalità.
        """
        if not pending:
            return weight * self._grid.manhattan(pos, self._grid.start_pos)
        nearest  = min(pending, key=lambda p: self._grid.manhattan(pos, p))
        farthest = max(pending, key=lambda p: self._grid.manhattan(pos, p))
        return weight * (
            self._grid.manhattan(pos, nearest)
            + self._grid.manhattan(farthest, self._grid.start_pos)
        )

    # ------------------------------------------------------------------
    # Stato iniziale e test goal
    # ------------------------------------------------------------------

    def _initial_state(self) -> SearchState:
        """
        Costruisce lo stato iniziale: drone alla posizione dei sensori
        con tutte le consegne pendenti.
        """
        pending = frozenset(
            pos for pos in self._grid.delivery_positions
            if self._grid[pos].has_pending_delivery()
        )
        return SearchState(
            position=self._sensors.position,
            pending_deliveries=pending,
            steps=0,
        )

    def _is_goal(self, state: SearchState) -> bool:
        """
        Verifica se lo stato è il goal:
        tutte le consegne effettuate (pending_deliveries=∅)
        e drone tornato alla base (position=start_pos).
        """
        return not state.pending_deliveries and state.position == self._grid.start_pos

    # ------------------------------------------------------------------
    # Successori
    # ------------------------------------------------------------------

    def _successors(
        self, node: SearchNode
    ) -> list[tuple[tuple[int, int], float, SearchState]]:
        """
        Genera i successori di un nodo espandendo le quattro direzioni cardinali.

        Per ogni direzione valida (in bounds e passabile) crea un nuovo SearchState:
        - aggiorna la posizione del drone
        - rimuove la cella dal pending_deliveries se è una DELIVERY

        Restituisce: lista di (nuova_posizione, costo_passo, nuovo_stato)
        """
        state, pos = node.state, node.state.position
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            new_pos = (pos[0] + dr, pos[1] + dc)
            r, c = new_pos
            if not self._grid.in_bounds(r, c) or not self._grid.is_passable(r, c):
                continue
            step_cost   = self._grid.step_cost(r, c)
            new_pending = set(state.pending_deliveries)
            new_pending.discard(new_pos)  # Se è una DELIVERY la rimuove
            new_state = SearchState(
                position=new_pos,
                pending_deliveries=frozenset(new_pending),
                steps=state.steps + 1,
            )
            result.append((new_pos, step_cost, new_state))
        return result

    # ------------------------------------------------------------------
    # Helper timer e costruzione risultato
    # ------------------------------------------------------------------

    @staticmethod
    def _timer() -> float:
        return time.perf_counter()

    def _make_result(self, algo, node, start_pos, expanded, max_f, t0):
        """Costruisce un SearchResult dal nodo goal trovato (o None se fallito)."""
        elapsed = self._timer() - t0
        if node is None:
            return SearchResult(
                algo, [], 0.0, expanded, max_f, elapsed, False,
                "nessun percorso trovato"
            )
        return SearchResult(
            algo, node.full_path(start_pos),
            node.g_cost, expanded, max_f, elapsed, True
        )

    # ------------------------------------------------------------------
    # BFS — Breadth-First Search
    # ------------------------------------------------------------------

    def bfs(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        root = SearchNode(start, None, None, 0.0)
        frontier = deque([root])  # FIFO: i nodi più superficiali vengono estratti prima
        visited  = {start}
        expanded, max_f = 0, 1

        while frontier:
            max_f = max(max_f, len(frontier))
            node = frontier.popleft()  # Estrae il nodo più superficiale
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("BFS", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                if new_state not in visited:
                    visited.add(new_state)  # Marcato prima dell'enqueue (corretto per BFS)
                    frontier.append(
                        SearchNode(new_state, node, new_pos,
                                   node.g_cost + cost, depth=node.depth + 1)
                    )
        return self._make_result("BFS", None, start.position, expanded, max_f, t0)

    # ------------------------------------------------------------------
    # DFS — Depth-First Search
    # ------------------------------------------------------------------

    def dfs(self, max_depth: int = 200) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        frontier = [SearchNode(start, None, None, 0.0)]  # Stack LIFO
        visited  = set()
        expanded, max_f = 0, 1

        while frontier:
            max_f = max(max_f, len(frontier))
            node = frontier.pop()  # LIFO: estrae il nodo più recentemente aggiunto
            if node.state in visited:
                continue
            visited.add(node.state)
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("DFS", node, start.position, expanded, max_f, t0)
            # Espande solo se non ha raggiunto il limite di profondità
            if node.depth < max_depth:
                for new_pos, cost, new_state in self._successors(node):
                    if new_state not in visited:
                        frontier.append(
                            SearchNode(new_state, node, new_pos,
                                       node.g_cost + cost, depth=node.depth + 1)
                        )
        return self._make_result("DFS", None, start.position, expanded, max_f, t0)

    # ------------------------------------------------------------------
    # IDS — Iterative Deepening Search
    # ------------------------------------------------------------------

    def ids(self, max_depth: int = 200) -> SearchResult:
        t0, start, total_exp = self._timer(), self._initial_state(), 0
        for limit in range(max_depth + 1):
            result, exp, _ = self._dls(start, limit)
            total_exp += exp
            if result is not None:
                return self._make_result("IDS", result, start.position, total_exp, 0, t0)
        return self._make_result("IDS", None, start.position, total_exp, 0, t0)

    def _dls(self, start: SearchState, limit: int):
        root  = SearchNode(start, None, None, 0.0)
        stack = [(root, set())]
        expanded, max_f = 0, 1

        while stack:
            max_f = max(max_f, len(stack))
            node, path_states = stack.pop()
            expanded += 1
            if self._is_goal(node.state):
                return node, expanded, max_f
            if node.depth < limit:
                for new_pos, cost, new_state in self._successors(node):
                    if new_state not in path_states:
                        stack.append((
                            SearchNode(new_state, node, new_pos,
                                       node.g_cost + cost, depth=node.depth + 1),
                            path_states | {new_state}
                        ))
        return None, expanded, max_f

    # ------------------------------------------------------------------
    # UCS — Uniform Cost Search
    # ------------------------------------------------------------------

    def ucs(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        frontier = [(0.0, SearchNode(start, None, None, 0.0))]
        visited  = {}  # stato -> miglior g(n) trovato
        expanded, max_f = 0, 1

        while frontier:
            max_f = max(max_f, len(frontier))
            g, node = heapq.heappop(frontier)  # Estrae il nodo a costo minore
            # Salta se abbiamo già trovato un percorso migliore per questo stato
            if node.state in visited and visited[node.state] <= g:
                continue
            visited[node.state] = g
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("UCS", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                new_g = node.g_cost + cost
                if new_state not in visited or visited[new_state] > new_g:
                    heapq.heappush(
                        frontier,
                        (new_g, SearchNode(new_state, node, new_pos, new_g,
                                           depth=node.depth + 1))
                    )
        return self._make_result("UCS", None, start.position, expanded, max_f, t0)

    # ------------------------------------------------------------------
    # Greedy Best-First Search
    # ------------------------------------------------------------------

    def greedy(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0 = self._heuristic(start.position, start.pending_deliveries)
        frontier = [(h0, SearchNode(start, None, None, 0.0, h0))]
        visited  = set()
        expanded, max_f = 0, 1

        while frontier:
            max_f = max(max_f, len(frontier))
            _, node = heapq.heappop(frontier)  # Ordina solo per h(n)
            if node.state in visited:
                continue
            visited.add(node.state)
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("Greedy", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                if new_state not in visited:
                    h = self._heuristic(new_state.position, new_state.pending_deliveries)
                    heapq.heappush(
                        frontier,
                        (h, SearchNode(new_state, node, new_pos,
                                       node.g_cost + cost, h, node.depth + 1))
                    )
        return self._make_result("Greedy", None, start.position, expanded, max_f, t0)

    # ------------------------------------------------------------------
    # A* — algoritmo principale
    # ------------------------------------------------------------------

    def astar(self, weight: float = 1.0) -> SearchResult:
        t0, start = self._timer(), self._initial_state() # FIX: stato iniziale corretto
        h0   = self._heuristic(start.position, start.pending_deliveries, weight) # FIX: euristica con peso
        root = SearchNode(start, None, None, 0.0, h0) # FIX: nodo radice con g=0 e h calcolata
        frontier = [(root.f_cost, root)] # frontiera ordinata per f(n) = g(n) + h(n)
        best_g   = {}  # stato -> miglior g(n) espanso
        expanded, max_f = 0, 1

        while frontier:
            max_f = max(max_f, len(frontier))
            _, node = heapq.heappop(frontier)
            # Graph search: salta se abbiamo già espanso con g(n) minore
            if node.state in best_g and best_g[node.state] < node.g_cost:
                continue
            best_g[node.state] = node.g_cost
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("A*", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                new_g = node.g_cost + cost
                if new_state not in best_g or best_g[new_state] > new_g:
                    h = self._heuristic(new_state.position,
                                        new_state.pending_deliveries, weight)
                    heapq.heappush(
                        frontier,
                        (new_g + h, SearchNode(new_state, node, new_pos,
                                               new_g, h, node.depth + 1))
                    )
        return self._make_result("A*", None, start.position, expanded, max_f, t0)

    # ------------------------------------------------------------------
    # IDA* — Iterative Deepening A*
    # ------------------------------------------------------------------

    def idastar(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0   = self._heuristic(start.position, start.pending_deliveries)
        root = SearchNode(start, None, None, 0.0, h0)
        threshold, total_exp = h0, 0

        while True:
            result, exp, next_t = self._ida_search(root, threshold, set())
            total_exp += exp
            if result is not None:
                return self._make_result("IDA*", result, start.position,
                                         total_exp, 0, t0)
            if next_t == float("inf"):
                return self._make_result("IDA*", None, start.position,
                                         total_exp, 0, t0)
            threshold = next_t  # Nuova soglia = min f(n) che ha superato la precedente

    def _ida_search(self, node: SearchNode, threshold: float, path_states: set):
       
        if node.f_cost > threshold:
            return None, 0, node.f_cost
        if self._is_goal(node.state):
            return node, 1, threshold
        min_exceeded, expanded = float("inf"), 1
        path_states = path_states | {node.state}
        for new_pos, cost, new_state in self._successors(node):
            if new_state in path_states:
                continue
            h     = self._heuristic(new_state.position, new_state.pending_deliveries)
            child = SearchNode(new_state, node, new_pos,
                               node.g_cost + cost, h, node.depth + 1)
            result, sub_exp, sub_t = self._ida_search(child, threshold, path_states)
            expanded += sub_exp
            if result is not None:
                return result, expanded, threshold
            min_exceeded = min(min_exceeded, sub_t)
        return None, expanded, min_exceeded

    # ------------------------------------------------------------------
    # RBFS — Recursive Best-First Search
    # ------------------------------------------------------------------

    def rbfs(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0   = self._heuristic(start.position, start.pending_deliveries)
        root = SearchNode(start, None, None, 0.0, h0)
        counter = [0]  # Lista mutabile per aggiornamento nelle chiamate ricorsive
        result, _ = self._rbfs_search(root, float("inf"), set(), counter)
        return self._make_result("RBFS", result, start.position, counter[0], 0, t0)

    def _rbfs_search(self, node: SearchNode, f_limit: float,
                     path_states: set, counter: list):
        """Ricerca ricorsiva per RBFS."""
        if self._is_goal(node.state):
            return node, node.f_cost
        successors = []
        path_states = path_states | {node.state}
        for new_pos, cost, new_state in self._successors(node):
            if new_state in path_states:
                continue
            h = self._heuristic(new_state.position, new_state.pending_deliveries)
            g = node.g_cost + cost
            # max(g+h, node.f_cost): propaga il costo del padre se maggiore
            successors.append((
                max(g + h, node.f_cost),
                SearchNode(new_state, node, new_pos, g, h, node.depth + 1)
            ))
        if not successors:
            return None, float("inf")
        while True:
            successors.sort(key=lambda x: x[0])
            best_f, best = successors[0]
            if best_f > f_limit:
                return None, best_f
            # Secondo nodo migliore come nuovo f_limit per la ricorsione
            alt_f = successors[1][0] if len(successors) > 1 else float("inf")
            counter[0] += 1
            result, best_f_new = self._rbfs_search(
                best, min(f_limit, alt_f), path_states, counter
            )
            successors[0] = (best_f_new, best)  # Aggiorna il costo del ramo esplorato
            if result is not None:
                return result, best_f_new

    # ------------------------------------------------------------------
    # SMA* — Simplified Memory-Bounded A*
    # ------------------------------------------------------------------

    def smastar(self, memory_limit: int = 500) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0      = self._heuristic(start.position, start.pending_deliveries)
        root    = SearchNode(start, None, None, 0.0, h0)
        counter = 0  # Tie-breaker per ordinamento stabile nella priority queue
        frontier   = [(root.f_cost, counter, root)]
        in_frontier = {start: root.f_cost}  # stato -> miglior f(n) in frontiera
        expanded, max_f = 0, 1

        while frontier:
            max_f = max(max_f, len(frontier))
            _, _, node = heapq.heappop(frontier)
            in_frontier.pop(node.state, None)
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("SMA*", node, start.position,
                                         expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                new_g = node.g_cost + cost
                h     = self._heuristic(new_state.position, new_state.pending_deliveries)
                f     = new_g + h
                # Salta se esiste già un percorso migliore per questo stato
                if new_state in in_frontier and in_frontier[new_state] <= f:
                    continue
                counter += 1
                child = SearchNode(new_state, node, new_pos, new_g, h, node.depth + 1)
                heapq.heappush(frontier, (f, counter, child))
                in_frontier[new_state] = f
                # Limite di memoria: rimuove il nodo con f(n) peggiore
                if len(frontier) > memory_limit:
                    worst_idx = max(range(len(frontier)), key=lambda i: frontier[i][0])
                    worst_state = frontier[worst_idx][2].state
                    frontier[worst_idx] = frontier[-1]
                    frontier.pop()
                    heapq.heapify(frontier)
                    in_frontier.pop(worst_state, None)
        return self._make_result("SMA*", None, start.position, expanded, max_f, t0)

    # ------------------------------------------------------------------
    # Confronto di tutti gli algoritmi abilitati
    # ------------------------------------------------------------------

    def compare_all(self, verbose: bool = True) -> dict[str, SearchResult]:
        """
        Esegue tutti gli algoritmi abilitati in ENABLED_ALGORITHMS e
        restituisce un dizionario {nome_algoritmo: SearchResult}.

        Prima di ogni run:
        - reset_deliveries(): reimposta le consegne pendenti.
        - sensors.reset(): riporta il drone alla posizione iniziale.
        Questo garantisce che ogni algoritmo parta dallo stesso stato iniziale.
        """
        algo_map = {
            "BFS":    self.bfs,
            "DFS":    self.dfs,
            "IDS":    self.ids,
            "UCS":    self.ucs,
            "Greedy": self.greedy,
            "A*":     self.astar,
            "IDA*":   self.idastar,
            "RBFS":   self.rbfs,
            "SMA*":   self.smastar,
        }
        results = {}
        for name, fn in algo_map.items():
            if not ENABLED_ALGORITHMS.get(name, False):
                if verbose:
                    print(f"[{name}] disabilitato")
                continue
            # Reset completo: consegne + posizione drone
            self._grid.reset_deliveries()
            self._sensors.reset(self._grid.start_pos)  # FIX: reset sensori
            try:
                res = fn()
            except RecursionError:
                res = SearchResult(name, [], 0.0, 0, 0, 0.0, False, "RecursionError")
            results[name] = res
            if verbose:
                print(res)
        return results
