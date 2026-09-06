
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

from environment import Grid, CellType
from image_parser import parse_image, generate_test_image
from sensors import DroneSensors
from movement import MovementEngine
from controller import SearchController, SearchResult, ENABLED_ALGORITHMS

try:
    import pygame
except ImportError:
    print("[main] pygame non trovato. Installa con:  pip install pygame")
    sys.exit(1)

# Costanti layout
WINDOW_W    = 1100
WINDOW_H    = 720
SIDEBAR_W   = 280
GRID_AREA_W = WINDOW_W - SIDEBAR_W
GRID_AREA_H = WINDOW_H
CELL_PAD    = 2
FPS         = 60

COLORS = { # Palette di colori per la visualizzazione
    CellType.EMPTY:    (240, 240, 235), 
    CellType.START:    ( 46, 125,  50),
    CellType.BLOCKED:  ( 55,  71,  79),
    CellType.DELIVERY: ( 21, 101, 192),
    CellType.WIND:     (  0, 172, 193),
    "bg":           ( 18,  18,  24),
    "sidebar_bg":   ( 26,  26,  36),
    "panel_bg":     ( 36,  36,  50),
    "text":         (220, 220, 230),
    "text_dim":     (120, 120, 140),
    "accent":       ( 99, 179, 237),
    "success":      ( 72, 199, 142),
    "warning":      (255, 193,   7),
    "danger":       (239,  68,  68),
    "drone":        (255, 255, 255),
    "path_planned": (150, 150, 255),
    "path_done":    ( 80, 200, 120),
}

ALGO_KEYS = {
    pygame.K_1: "BFS", pygame.K_2: "DFS",    pygame.K_3: "IDS",
    pygame.K_4: "UCS", pygame.K_5: "Greedy", pygame.K_6: "A*",
    pygame.K_7: "IDA*",pygame.K_8: "RBFS",   pygame.K_9: "SMA*",
}



# Stato simulazione

class SimState: 
    def __init__(self, grid: Grid, path: list, result: SearchResult, sensors: DroneSensors) -> None:
        self.grid    = grid
        self.full_path = path
        self.result  = result
        self.sensors = sensors
        self.step_index = 0
        self.anim_t     = 0.0
        self.paused     = True
        self.finished   = False
        self.log: list[str] = []

    @property # Progressione della simulazione (0.0-1.0)
    def progress(self) -> float:
        return self.step_index / max(1, len(self.full_path) - 1)

    def current_cell(self) -> tuple[int, int]:
        return self.full_path[self.step_index] if self.full_path else self.grid.start_pos

    def next_cell(self) -> Optional[tuple[int, int]]:
        if self.step_index + 1 < len(self.full_path):
            return self.full_path[self.step_index + 1]
        return None


# Renderer Pygame per visualizzare la griglia, il drone e la sidebar con le informazioni.
class Renderer:
    def __init__(self, screen, font_sm, font_md, font_lg):
        self.screen  = screen
        self.font_sm = font_sm
        self.font_md = font_md
        self.font_lg = font_lg

    def draw_frame(self, sim: SimState, algo_name: str, speed: float, cell_size: int, ox: int, oy: int):
        self.screen.fill(COLORS["bg"])
        self._draw_grid(sim, cell_size, ox, oy)
        self._draw_path(sim, cell_size, ox, oy)
        self._draw_drone(sim, cell_size, ox, oy)
        self._draw_sidebar(sim, algo_name, speed)
        pygame.display.flip()

    # --- Griglia ---
    def _draw_grid(self, sim, cell_size, ox, oy):
        for r in range(sim.grid.rows):
            for c in range(sim.grid.cols):
                cell  = sim.grid[r, c]
                color = COLORS[cell.cell_type]
                if cell.delivered:
                    color = self._blend(color, COLORS["success"], 0.35)
                x = ox + c * cell_size + CELL_PAD
                y = oy + r * cell_size + CELL_PAD
                w = h = cell_size - CELL_PAD * 2
                pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=3)
                label = {CellType.START:"S", CellType.BLOCKED:"X",
                         CellType.DELIVERY:"D", CellType.WIND:"W"}.get(cell.cell_type, "")
                if label and cell_size >= 20:
                    dark = cell.cell_type == CellType.WIND
                    surf = self.font_sm.render(label, True, (30,30,30) if dark else (255,255,255))
                    self.screen.blit(surf, surf.get_rect(center=(x+w//2, y+h//2)))

    # --- Percorso ---
    def _draw_path(self, sim, cell_size, ox, oy):
        path = sim.full_path
        if len(path) < 2:
            return
        half = cell_size // 2
        surf = pygame.Surface((GRID_AREA_W, GRID_AREA_H), pygame.SRCALPHA)
        for i in range(1, len(path)):
            r0, c0 = path[i-1]
            r1, c1 = path[i]
            color = (*COLORS["path_done"], 160) if i <= sim.step_index else (*COLORS["path_planned"], 80)
            pygame.draw.line(surf, color,
                             (c0*cell_size+half, r0*cell_size+half),
                             (c1*cell_size+half, r1*cell_size+half), 2)
        self.screen.blit(surf, (ox, oy))

    # --- Drone ---
    def _draw_drone(self, sim, cell_size, ox, oy):
        half = cell_size // 2
        curr = sim.current_cell()
        nxt  = sim.next_cell()
        if nxt and sim.anim_t > 0:
            t  = sim.anim_t
            px = ox + (curr[1]*(1-t) + nxt[1]*t) * cell_size + half
            py = oy + (curr[0]*(1-t) + nxt[0]*t) * cell_size + half
        else:
            px = ox + curr[1]*cell_size + half
            py = oy + curr[0]*cell_size + half
        r = max(4, cell_size // 3)
        pygame.draw.circle(self.screen, COLORS["drone"], (int(px), int(py)), r)
        pygame.draw.circle(self.screen, COLORS["accent"], (int(px), int(py)), max(2, r-3))
        for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            ex = int(px + dx*(r+4)*0.7)
            ey = int(py + dy*(r+4)*0.7)
            pygame.draw.line(self.screen, COLORS["text_dim"], (int(px), int(py)), (ex, ey), 1)
            pygame.draw.circle(self.screen, COLORS["text_dim"], (ex, ey), 2)

    # --- Sidebar ---
    def _draw_sidebar(self, sim, algo_name, speed):
        sx = GRID_AREA_W
        pygame.draw.rect(self.screen, COLORS["sidebar_bg"], (sx, 0, SIDEBAR_W, WINDOW_H))
        y, pad = 16, 14

        y = self._text("SMART DRONE", sx+pad, y, self.font_lg, COLORS["accent"])
        y = self._text("DELIVERY",    sx+pad, y, self.font_lg, COLORS["accent"])
        y += 8
        pygame.draw.line(self.screen, COLORS["panel_bg"], (sx, y), (WINDOW_W, y), 1)
        y += 10

        y = self._section(sx, y, pad, "ALGORITMO")
        y = self._text(algo_name, sx+pad, y, self.font_md, COLORS["text"])
        if sim.result.success:
            y = self._text(f"Nodi:   {sim.result.nodes_expanded}", sx+pad, y, self.font_sm, COLORS["text_dim"])
            y = self._text(f"Tempo:  {sim.result.time_seconds*1000:.1f} ms", sx+pad, y, self.font_sm, COLORS["text_dim"])
            y = self._text(f"Costo:  {sim.result.total_cost:.1f}", sx+pad, y, self.font_sm, COLORS["text_dim"])
            y = self._text(f"Passi:  {len(sim.full_path)-1}", sx+pad, y, self.font_sm, COLORS["text_dim"])
        else:
            y = self._text("Nessun percorso trovato", sx+pad, y, self.font_sm, COLORS["danger"])
        y += 8

        y = self._section(sx, y, pad, "DRONE")
        y = self._text(f"Posizione: {sim.sensors.position}", sx+pad, y, self.font_sm, COLORS["text_dim"])
        y = self._text(f"Passi:     {sim.sensors.steps_taken}", sx+pad, y, self.font_sm, COLORS["text_dim"])
        y += 8

        total   = len(sim.grid.delivery_positions)
        pending = len(sim.grid.pending_deliveries())
        done    = total - pending
        y = self._section(sx, y, pad, "CONSEGNE")
        y = self._text(f"{done}/{total} completate", sx+pad, y, self.font_md,
                       COLORS["success"] if done == total else COLORS["text"])
        for pos in sim.grid.delivery_positions:
            cell  = sim.grid[pos]
            icon  = "v" if cell.delivered else "."
            color = COLORS["success"] if cell.delivered else COLORS["text_dim"]
            y = self._text(f"  {icon} {pos}", sx+pad, y, self.font_sm, color)
        y += 8

        y = self._section(sx, y, pad, "SIMULAZIONE")
        if sim.finished:
            status, sc = "COMPLETATA", COLORS["success"]
        elif not sim.paused:
            status, sc = "IN CORSO",   COLORS["accent"]
        else:
            status, sc = "IN PAUSA",   COLORS["text_dim"]
        y = self._text(status, sx+pad, y, self.font_sm, sc)
        y = self._bar(sx+pad, y, SIDEBAR_W-pad*2, 10, sim.progress, COLORS["accent"])
        y = self._text(f"Velocità: {speed:.1f}x", sx+pad, y+4, self.font_sm, COLORS["text_dim"])
        y += 8

        if sim.log:
            y = self._section(sx, y, pad, "LOG")
            for msg in sim.log[-6:]:
                y = self._text(msg, sx+pad, y, self.font_sm, COLORS["text_dim"])

        y = WINDOW_H - 130
        pygame.draw.line(self.screen, COLORS["panel_bg"], (sx, y), (WINDOW_W, y), 1)
        y += 8
        for ctrl in ["SPACE  pausa/avvia", "R      ripristina",
                     "1-9    algoritmo", "+/-    velocità",
                     "C      confronta", "M      nuova mappa", "ESC/Q  esci"]:
            y = self._text(ctrl, sx+pad, y, self.font_sm, COLORS["text_dim"])

    # --- Helpers ---
    def _text(self, text, x, y, font, color) -> int:
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 2

    def _section(self, sx, y, pad, title) -> int:
        pygame.draw.rect(self.screen, COLORS["panel_bg"],
                         (sx+pad-4, y, SIDEBAR_W-pad*2+8, 18), border_radius=3)
        return self._text(title, sx+pad, y+1, self.font_sm, COLORS["accent"]) + 4

    def _bar(self, x, y, w, h, frac, color) -> int:
        frac = max(0.0, min(1.0, frac))
        pygame.draw.rect(self.screen, COLORS["panel_bg"], (x, y, w, h), border_radius=3)
        if frac > 0:
            pygame.draw.rect(self.screen, color, (x, y, int(w*frac), h), border_radius=3)
        return y + h

    @staticmethod
    def _blend(c1, c2, t):
        return tuple(int(a*(1-t)+b*t) for a, b in zip(c1, c2))


# Funzioni di supporto
def compute_cell_size(grid: Grid):
    cs = max(8, min(GRID_AREA_W // grid.cols, GRID_AREA_H // grid.rows))
    return cs, (GRID_AREA_W - cs*grid.cols)//2, (GRID_AREA_H - cs*grid.rows)//2


TIMEOUT_SECONDS = 300  # Timeout per algoritmi lenti (5 minuti)

def build_sim(grid: Grid, algo_name: str) -> SimState:
    import threading
    from controller import SearchResult

    grid.reset_deliveries()
    sensors    = DroneSensors(grid, grid.start_pos)
    controller = SearchController(grid, sensors)
    fn = {"BFS":controller.bfs,"DFS":controller.dfs,"IDS":controller.ids,
          "UCS":controller.ucs,"Greedy":controller.greedy,"A*":controller.astar,
          "IDA*":controller.idastar,"RBFS":controller.rbfs,"SMA*":controller.smastar}[algo_name]

    result_box = [None]

    def run():
        try:
            result_box[0] = fn()
        except RecursionError:
            result_box[0] = SearchResult(algo_name, [], 0.0, 0, 0, 0.0, False, "RecursionError")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=TIMEOUT_SECONDS)

    TIMEOUT_MSG = {
        "IDS":    "IDS esplora per profondita' crescente: con piu' consegne lo spazio degli stati esplode.",
        "Greedy": "Greedy non considera il costo reale: puo' esplorare percorsi molto lunghi.",
        "IDA*":   "IDA* e' ottimale ma ricalcola molti nodi: troppo lento su problemi multi-consegna.",
        "RBFS":   "RBFS usa memoria lineare ma rigenera nodi continuamente: non adatto a questo dominio.",
        "SMA*":   "SMA* con memoria limitata cicla tra troppi nodi: il limite di memoria e' insufficiente per questo dominio.",
    }
    if result_box[0] is None:
        msg = TIMEOUT_MSG.get(algo_name, f"Timeout ({TIMEOUT_SECONDS}s)")
        result = SearchResult(algo_name, [], 0.0, 0, 0, TIMEOUT_SECONDS, False, msg)
    else:
        result = result_box[0]

    path    = result.path if result.success else [grid.start_pos]
    sensors2 = DroneSensors(grid, grid.start_pos)
    sim = SimState(grid, path, result, sensors2)
    sim.log.append(f"Algoritmo: {algo_name}")
    sim.log.append(f"Passi: {len(path)-1}" if result.success else f"ERRORE: {result.message}")
    return sim


def advance_sim(sim: SimState, dt: float, speed: float):
    if sim.paused or sim.finished:
        return
    sim.anim_t += dt * speed
    if sim.anim_t >= 1.0:
        sim.anim_t = 0.0
        if sim.step_index + 1 < len(sim.full_path):
            sim.step_index += 1
            new_pos = sim.full_path[sim.step_index]
            sim.sensors.update(new_pos)
            r, c = new_pos
            cell = sim.grid[r, c]
            if cell.has_pending_delivery():
                if sim.grid.mark_delivered(r, c):
                    sim.log.append(f"Consegna {new_pos} OK")
            if cell.cell_type == CellType.WIND:
                sim.log.append(f"Vento in {new_pos} - costo x2")
        else:
            sim.finished = True
            sim.log.append("Simulazione completata!")


def draw_loading(screen, font_lg, font_sm, msg, error=False):
    screen.fill((18, 18, 24))
    w, h = screen.get_size()
    surf = font_lg.render("SMART DRONE DELIVERY", True, (99, 179, 237))
    screen.blit(surf, surf.get_rect(center=(w//2, h//2-50)))
    color = (239, 68, 68) if error else (120, 120, 140)
    # Wrap long messages across multiple lines
    words = msg.split()
    lines, line = [], []
    for word in words:
        test = " ".join(line + [word])
        if font_sm.size(test)[0] > w - 120:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    y_start = h//2 if not error else h//2 - 10
    for i, l in enumerate(lines):
        surf2 = font_sm.render(l, True, color)
        screen.blit(surf2, surf2.get_rect(center=(w//2, y_start + i*20)))
    if error:
        hint = font_sm.render("Premi R per tornare ad A* | 6 per A*", True, (99, 179, 237))
        screen.blit(hint, hint.get_rect(center=(w//2, y_start + len(lines)*20 + 20)))
    pygame.display.flip()


# Loop principale

def main(image_path: Optional[str] = None):
    sys.setrecursionlimit(50000)  # IDA* e RBFS sono ricorsivi
    if image_path and Path(image_path).exists():
        grid = parse_image(image_path)
    else:
        print("[main] Nessuna immagine fornita — uso la mappa di test.")
        grid = parse_image(generate_test_image("test_map.png"))

    pygame.init()
    pygame.display.set_caption("Smart Drone Delivery")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock  = pygame.time.Clock()

    try:
        font_sm = pygame.font.SysFont("consolas", 13)
        font_md = pygame.font.SysFont("consolas", 16, bold=True)
        font_lg = pygame.font.SysFont("consolas", 22, bold=True)
    except Exception:
        font_sm = pygame.font.Font(None, 14)
        font_md = pygame.font.Font(None, 18)
        font_lg = pygame.font.Font(None, 24)

    renderer  = Renderer(screen, font_sm, font_md, font_lg)
    cell_size, ox, oy = compute_cell_size(grid)

    algo_name = "A*"
    speed     = 2.0

    draw_loading(screen, font_lg, font_sm, f"Calcolo percorso con {algo_name}...")
    pygame.event.pump()
    sim = build_sim(grid, algo_name)
    if not sim.result.success:
        draw_loading(screen, font_lg, font_sm, sim.result.message, error=True)
        pygame.event.pump()
        pygame.time.wait(4000)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif key == pygame.K_SPACE:
                    if sim.finished:
                        draw_loading(screen, font_lg, font_sm, f"Calcolo con {algo_name}...")
                        pygame.event.pump()
                        sim = build_sim(grid, algo_name)
                    else:
                        sim.paused = not sim.paused
                elif key == pygame.K_r:
                    draw_loading(screen, font_lg, font_sm, f"Calcolo con {algo_name}...")
                    pygame.event.pump()
                    sim = build_sim(grid, algo_name)
                    if not sim.result.success:
                        draw_loading(screen, font_lg, font_sm, sim.result.message, error=True)
                        pygame.event.pump()
                        pygame.time.wait(4000)
                elif key in ALGO_KEYS:
                    algo_name = ALGO_KEYS[key]
                    draw_loading(screen, font_lg, font_sm, f"Calcolo con {algo_name}...")
                    pygame.event.pump()
                    sim = build_sim(grid, algo_name)
                    if not sim.result.success:
                        draw_loading(screen, font_lg, font_sm, sim.result.message, error=True)
                        pygame.event.pump()
                        pygame.time.wait(4000)
                elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(20.0, speed + 0.5)
                elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(0.5, speed - 0.5)
                elif key == pygame.K_c:
                    print("\n" + "="*55)
                    print("CONFRONTO ALGORITMI ABILITATI")
                    print("="*55)
                    SearchController(grid, DroneSensors(grid, grid.start_pos)).compare_all(verbose=True)
                    print("="*55 + "\n")
                    grid.reset_deliveries()
                elif key == pygame.K_m:
                    import random
                    from image_parser import _generate_random_layout
                    draw_loading(screen, font_lg, font_sm, "Generazione nuova mappa...")
                    pygame.event.pump()
                    layout = _generate_random_layout(seed=random.randint(0, 1000000))
                    generate_test_image("test_map.png", layout=layout)
                    grid = parse_image("test_map.png")
                    cell_size, ox, oy = compute_cell_size(grid)
                    sim = build_sim(grid, algo_name)
                    if not sim.result.success:
                        draw_loading(screen, font_lg, font_sm, sim.result.message, error=True)
                        pygame.event.pump()
                        pygame.time.wait(4000)

        advance_sim(sim, dt, speed)
        renderer.draw_frame(sim, algo_name, speed, cell_size, ox, oy)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Drone Delivery")
    parser.add_argument("image", nargs="?", default=None)
    args = parser.parse_args()
    main(args.image)