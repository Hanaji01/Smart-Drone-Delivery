

from __future__ import annotations

from pathlib import Path
import random
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from environment import CellType, Grid, GridCell


# Mappatura simbolo -> CellType (usata dal classificatore ML)

SYMBOL_TO_CELL: dict[str, CellType] = {
    "S": CellType.START,
    "D": CellType.DELIVERY,
    "W": CellType.WIND,
    "X": CellType.BLOCKED,
    "E": CellType.EMPTY,
}

CELL_SIZE = 28   # Dimensione celle in pixel (compatibile con il dataset ML)



# Parsing con classificatore ML

def parse_image_ml(
    image_path: str | Path,
    cell_size: int = CELL_SIZE,
) -> Grid:

    from ml_classifier import load_model, predict_cell

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    # Carica il modello CNN addestrato (pesi PyTorch)
    model = load_model()

    # Apre l'immagine in scala di grigi (come il dataset di training)
    img     = Image.open(image_path).convert("L")
    width, height = img.size
    img_arr = np.array(img)

    # Calcola il numero di righe e colonne della griglia
    n_rows = height // cell_size
    n_cols = width  // cell_size

    if n_rows == 0 or n_cols == 0:
        raise ValueError(
            f"Immagine troppo piccola ({width}x{height}) per celle {cell_size}x{cell_size}."
        )

    cells = np.empty((n_rows, n_cols), dtype=object)
    symbols_predicted = []

    for row in range(n_rows):
        for col in range(n_cols):
            # Ritaglia la cella dall'immagine
            y0 = row * cell_size
            x0 = col * cell_size
            cell_img = img_arr[y0:y0 + cell_size, x0:x0 + cell_size]

            # Classifica la cella con la CNN
            symbol    = predict_cell(cell_img, model, cell_size)
            symbols_predicted.append(symbol)
            cell_type = SYMBOL_TO_CELL.get(symbol, CellType.EMPTY)
            cells[row, col] = GridCell(cell_type=cell_type)

    grid = Grid(cells)
    print(f"[image_parser ML] Griglia {grid.rows}x{grid.cols} caricata da '{image_path.name}'")
    print(f"  Simboli predetti: {''.join(symbols_predicted)}")
    print(f"  Modalita': CNN (PyTorch, 2 layer convoluzionali)")
    print(f"  START={grid.start_pos}")
    print(f"  Consegne trovate: {len(grid.delivery_positions)}")
    return grid



# Funzione principale — parsing ML

def parse_image(
    image_path: str | Path,
    cell_size: int = CELL_SIZE,
) -> Grid:
    """Punto di accesso principale per il parsing della mappa usando solo ML."""
    return parse_image_ml(image_path, cell_size)



# Generazione immagine di test compatibile con ML

def generate_test_image(
    output_path: str | Path = "test_map.png",
    layout: Optional[list] = None,
    cell_size: int = CELL_SIZE,
) -> Path:
    """
    Genera un'immagine di test con i simboli in celle 28x28.
    Compatibile con il parser ML.

    Ogni cella e' un quadrato cell_size x cell_size con la lettera
    del simbolo centrata — lo stesso formato usato nel dataset di training.
    """
    if layout is None:
        layout = [
            "SDEWXEEE",
            "EXXEEXEE",
            "EEXEEEED",
            "EEEWWEEE",
            "EWEWEWEE",
            "EEEEEEEE",
            "EEEDEXEE",
            "EEEEEEEE",
        ]

    rows   = len(layout)
    cols   = max(len(r) for r in layout)
    width  = cols * cell_size
    height = rows * cell_size

    # Immagine in scala di grigi — sfondo bianco, lettere nere.
    # Stesso formato del dataset di training: il KNN classifica
    # la forma del simbolo, non il colore di sfondo.
    img  = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except (IOError, OSError):
        try:
            # Windows path
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()

    for r, row_str in enumerate(layout):
        for c, ch in enumerate(row_str):
            sym = ch.upper()
            if sym not in SYMBOL_TO_CELL:
                sym = "E"

            x0, y0 = c * cell_size, r * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size

            # Sfondo bianco (come il dataset di training)
            draw.rectangle([x0, y0, x1, y1], fill=255)
            # Bordo sottile per delimitare le celle
            draw.rectangle([x0, y0, x1, y1], outline=180, width=1)

            # Lettera nera centrata (come il dataset di training)
            bbox = draw.textbbox((0, 0), sym, font=font)
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]
            tx   = x0 + (cell_size - tw) // 2
            ty   = y0 + (cell_size - th) // 2
            draw.text((tx, ty), sym, fill=0, font=font)

    output_path = Path(output_path)
    img.save(output_path)
    print(f"[generate_test_image] Immagine salvata in '{output_path}' "
          f"({cols}x{rows} celle, {cell_size}px per cella)")
    return output_path



# Generazione di piu' mappe di test con layout casuali

def _generate_random_layout(
    rows: int = 8,
    cols: int = 8,
    n_delivery: int = 2,
    n_blocked: int = 4,
    n_wind: int = 4,
    seed: Optional[int] = None,
) -> list[str]:
    """
    Genera un layout casuale valido: una sola cella START (S), una o piu'
    DELIVERY (D), alcune BLOCKED (X) e WIND (W), il resto EMPTY (E).
    """
    rng = random.Random(seed)

    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(all_positions)

    # La prima posizione estratta e' sempre lo START, cosi' ce n'e' sempre
    # esattamente una per griglia (requisito del simulatore).
    start_pos = all_positions[0]
    delivery_positions = all_positions[1:1 + n_delivery]
    blocked_positions   = all_positions[1 + n_delivery: 1 + n_delivery + n_blocked]
    wind_positions       = all_positions[1 + n_delivery + n_blocked: 1 + n_delivery + n_blocked + n_wind]

    grid_chars = [["E"] * cols for _ in range(rows)]
    r, c = start_pos
    grid_chars[r][c] = "S"
    for r, c in delivery_positions:
        grid_chars[r][c] = "D"
    for r, c in blocked_positions:
        grid_chars[r][c] = "X"
    for r, c in wind_positions:
        grid_chars[r][c] = "W"

    return ["".join(row) for row in grid_chars]


def generate_multiple_test_maps(
    output_dir: str | Path = ".",
    n_maps: int = 5,
    rows: int = 8,
    cols: int = 8,
    cell_size: int = CELL_SIZE,
    seed: Optional[int] = 42,
) -> list[Path]:
    """
    Genera 'n_maps' immagini di test (test_map_1.png, test_map_2.png, ...)
    con layout casuali diversi tra loro, cosi' da poter valutare il parser
    ML su piu' mappe invece che su una sola.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    paths = []

    for i in range(1, n_maps + 1):
        layout = _generate_random_layout(rows=rows, cols=cols, seed=rng.randint(0, 1_000_000))
        out_path = output_dir / f"test_map_{i}.png"
        generate_test_image(out_path, layout=layout, cell_size=cell_size)
        paths.append(out_path)

    print(f"\n[generate_multiple_test_maps] Generate {n_maps} mappe di test in '{output_dir}'")
    return paths


if __name__ == "__main__":
    # Genera piu' mappe di test con layout diversi
    test_maps = generate_multiple_test_maps(n_maps=5)

    # Effettua il parsing ML di ciascuna mappa e mostra il risultato
    for i, test_img in enumerate(test_maps, start=1):
        print(f"\n{'=' * 50}")
        print(f"Mappa di test {i}: '{test_img.name}'")
        print("=" * 50)
        grid = parse_image(test_img)
        print("\nRappresentazione ASCII della griglia:")
        print(grid.ascii_render())