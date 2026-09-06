
from __future__ import annotations #

import random # Per generare perturbazioni casuali durante la creazione del dataset
import sys # Per uscire in caso di errori critici (es. PyTorch non installato)
from pathlib import Path # Per gestire i percorsi dei file in modo portabile
from typing import TYPE_CHECKING, Any, Optional # Per tipi opzionali e controllo dei tipi durante lo sviluppo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Variabili per gestione dell'import dinamico di PyTorch
_TORCH_IMPORT_ERROR: Optional[BaseException] = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    _TORCH_AVAILABLE = True
else:
    # Rete neurale (CNN) con PyTorch
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
        _TORCH_AVAILABLE = True
    except ImportError as exc:
        torch: Any = None
        nn: Any = None
        DataLoader: Any = None
        TensorDataset: Any = None
        train_test_split: Any = None
        accuracy_score: Any = None
        precision_score: Any = None
        recall_score: Any = None
        f1_score: Any = None
        classification_report: Any = None
        confusion_matrix: Any = None
        _TORCH_AVAILABLE = False
        _TORCH_IMPORT_ERROR = exc


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch (e/o scikit-learn per lo split/metriche) non è installato. "
            "Installa i pacchetti con: pip install torch scikit-learn"
        ) from _TORCH_IMPORT_ERROR


# Configurazione

CELL_SIZE     = 28        # Dimensione di ogni cella in pixel (come MNIST)
SAMPLES_CLASS = 1000       # Numero di campioni per classe nel dataset (aumentato da 200 a 1000)
DATASET_DIR   = Path("dataset")   # Cartella del dataset generato
MODEL_PATH    = Path("cnn_model.pt")  # Percorso dove salvare i pesi della CNN
EPOCHS        = 10        # Numero di epoche di addestramento
BATCH_SIZE    = 32        # Dimensione del batch
LEARNING_RATE = 1e-3      # Learning rate per l'optimizer Adam

# Simboli riconosciuti dal classificatore e loro etichetta di classe.
# L'ordine di questo dizionario definisce anche l'indice di classe usato
# dalla CNN (0=S, 1=D, 2=W, 3=X, 4=E) — deve restare fisso e coerente
# sia in fase di training che in fase di predizione.
SYMBOLS = {
    "S": "START",
    "D": "DELIVERY",
    "W": "WIND",
    "X": "BLOCKED",
    "E": "EMPTY",
}
CLASS_NAMES = list(SYMBOLS.keys())              # ["S", "D", "W", "X", "E"]
SYMBOL_TO_IDX = {s: i for i, s in enumerate(CLASS_NAMES)}
IDX_TO_SYMBOL = {i: s for i, s in enumerate(CLASS_NAMES)}


# 1. Definizione della rete neurale (CNN)

class CNN(nn.Module):
    """
    CNN per la classificazione di simboli 28x28 a canale singolo (scala di grigi).

    Architettura:
      Conv(1->16, 3x3) -> ReLU -> MaxPool(2x2)   # 28x28 -> 14x14
      Conv(16->32, 3x3) -> ReLU -> MaxPool(2x2)  # 14x14 -> 7x7
      Flatten -> Linear(32*7*7 -> 128) -> ReLU
      Linear(128 -> 5)                            # 5 classi di output (S,D,W,X,E)
    """

    def __init__(self, num_classes: int = len(SYMBOLS)):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # 28x28 -> 14x14

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: tensore (Batch, 1, 28, 28)
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x   # logits grezzi (CrossEntropyLoss applica internamente il softmax)


# 2. Generazione del Dataset

def _render_symbol(
    symbol: str,
    size: int = CELL_SIZE,
    angle: float = 0.0,
    add_noise: bool = False,
) -> np.ndarray:
    # Sfondo bianco, testo nero (come MNIST)
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    # Font fisso 16 — stesso usato da generate_test_image
    font_size = 16
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Centrato
    bbox  = draw.textbbox((0, 0), symbol, font=font)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]
    x     = (size - tw) // 2
    y     = (size - th) // 2
    draw.rectangle([0, 0, size, size], fill=255)
    draw.rectangle([0, 0, size, size], outline=180, width=1)
    draw.text((x, y), symbol, fill=0, font=font)

    # Augmentazione: piccola rotazione casuale, cosi' campioni diversi
    # della stessa classe non sono piu' pixel-identici. Necessario ora che
    # generate_dataset() produce molti piu' campioni per classe: senza
    # variabilita' aumentare il numero di immagini non aggiungerebbe
    # alcuna informazione utile all'addestramento.
    if angle != 0.0:
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

    arr = np.array(img, dtype=np.float32)

    # Augmentazione: leggero rumore gaussiano sui pixel, per simulare
    # imperfezioni del rendering/scansione e rendere il modello piu' robusto.
    if add_noise:
        noise = np.random.normal(loc=0.0, scale=8.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255)

    arr = arr / 255.0
    return arr.flatten()


def generate_dataset(
    dataset_dir: Path = DATASET_DIR,
    samples_per_class: int = SAMPLES_CLASS,
    cell_size: int = CELL_SIZE,
    max_rotation_deg: float = 12.0,
) -> None:
    """
    Genera il dataset di training: per ogni simbolo crea
    'samples_per_class' immagini con piccole rotazioni casuali e rumore,
    cosi' da avere campioni realmente diversi tra loro (necessario per un
    dataset di 1000 campioni per classe, non solo 200 duplicati identici).

    Struttura risultante:
      dataset/S/  -> 1000 immagini .png del simbolo S
      dataset/D/  -> 1000 immagini .png del simbolo D
      ...
    """
    print(f"[ML] Generazione dataset in '{dataset_dir}' ...")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        class_dir = dataset_dir / symbol
        class_dir.mkdir(exist_ok=True)

        for i in range(samples_per_class):
            # Il primo campione resta "canonico" (nessuna perturbazione),
            # gli altri vengono ruotati/rumorosi leggermente per varieta'.
            if i == 0:
                angle, noise = 0.0, False
            else:
                angle = random.uniform(-max_rotation_deg, max_rotation_deg)
                noise = True

            arr = _render_symbol(symbol, cell_size, angle=angle, add_noise=noise)
            # Ricostruisce immagine 28x28 per salvarla su disco
            img_arr = (arr.reshape(cell_size, cell_size) * 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="L")
            img.save(class_dir / f"{symbol}_{i:04d}.png")

        print(f"  {symbol} ({SYMBOLS[symbol]}): {samples_per_class} campioni generati")

    print(f"[ML] Dataset generato: {len(SYMBOLS)} classi x {samples_per_class} = "
          f"{len(SYMBOLS) * samples_per_class} immagini totali")



# 3. Caricamento Dataset

def load_dataset(dataset_dir: Path = DATASET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica le immagini dal dataset e le restituisce come:
      X: matrice (n_samples, 28, 28) — immagini in scala di grigi normalizzate,
         SENZA flatten (la CNN richiede la forma bidimensionale della cella)
      y: array (n_samples,)          — etichette di classe intere (indice in CLASS_NAMES)
    """
    X, y = [], []

    for symbol in SYMBOLS:
        class_dir = dataset_dir / symbol
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.png"):
            img = Image.open(img_path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr)                      # shape (28, 28), niente flatten
            y.append(SYMBOL_TO_IDX[symbol])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# 4. Training e Valutazione della CNN

def train_cnn(
    dataset_dir: Path = DATASET_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    test_size: float = 0.2,
    model_path: Path = MODEL_PATH,
) -> "CNN":

    _require_torch()

    print(f"\n[ML] Caricamento dataset da '{dataset_dir}' ...")
    X, y = load_dataset(dataset_dir)
    print(f"[ML] Dataset: {len(X)} campioni, {len(np.unique(y))} classi")

    # Train/test split (80% training, 20% test) — valutazione del modello
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[ML] Split: {len(X_train)} training, {len(X_test)} test")

    # Conversione in Tensori PyTorch mantenendo la shape (Batch, 1, 28, 28):
    # aggiungiamo la dimensione del canale (1 = scala di grigi) con unsqueeze.
    X_train_t = torch.from_numpy(X_train).unsqueeze(1)   # (N, 1, 28, 28)
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t  = torch.from_numpy(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(SYMBOLS)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n[ML] Addestramento CNN (epoche={epochs}, batch={batch_size}, lr={lr}) "
          f"su device='{device}' ...")

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"  Epoca {epoch}/{epochs} — loss: {epoch_loss:.4f} — accuracy training: {epoch_acc:.4f}")

    # Valutazione sul test set
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_t.to(device)
        outputs = model(X_test_dev)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    y_true = y_test

    print("\n[ML] ===== RISULTATI CLASSIFICAZIONE (CNN) =====")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F-measure: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    print("\n[ML] Report per classe:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("[ML] Matrice di confusione:")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))))

    # Salva SOLO i pesi (state_dict) — modo nativo/consigliato da PyTorch
    # per salvare un modello, invece del pickle usato dal KNN.
    torch.save(model.state_dict(), model_path)
    print(f"\n[ML] Pesi del modello salvati in '{model_path}'")

    return model



# 5. Predizione su singola cella

def load_model(model_path: Path = MODEL_PATH) -> "CNN":
    """Istanzia la CNN, carica i pesi salvati e imposta il modello in eval mode."""
    _require_torch()
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Modello non trovato in '{model_path}'. "
            f"Esegui prima train_cnn() o ml_classifier.py direttamente."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(SYMBOLS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()   # Modalita' valutazione: disattiva dropout/batchnorm in training mode
    return model


def predict_cell(
    cell_img: np.ndarray,
    model: "CNN",
    cell_size: int = CELL_SIZE,
) -> str:

    _require_torch()

    # Ridimensiona a 28x28 se necessario
    if cell_img.shape != (cell_size, cell_size):
        img = Image.fromarray(cell_img.astype(np.uint8)).resize(
            (cell_size, cell_size), Image.LANCZOS
        )
        cell_img = np.array(img)

    # Normalizza come durante il training (senza flatten: la CNN vuole 2D)
    arr = (cell_img.astype(np.float32) / 255.0)

    device = next(model.parameters()).device
    # unsqueeze(0) due volte: aggiunge dimensione canale e dimensione batch
    # arr (28,28) -> tensore (1, 1, 28, 28) = (Batch=1, Canale=1, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred_idx = output.argmax(dim=1).item()

    return IDX_TO_SYMBOL[pred_idx]


# -----------------------------------------------------------------------
# Entry point — eseguire questo file per generare il dataset e addestrare
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("Smart Drone Delivery — Componente ML")
    print("Classificatore CNN (PyTorch) per riconoscimento simboli")
    print("=" * 50)

    # Step 1: Genera il dataset se non esiste
    if not DATASET_DIR.exists():
        generate_dataset()
    else:
        print(f"[ML] Dataset esistente trovato in '{DATASET_DIR}' — skip generazione")

    # Step 2: Addestra la CNN e valuta le prestazioni
    if not _TORCH_AVAILABLE:
        print("\n[ML] Errore: PyTorch e/o scikit-learn non sono installati.")
        print("Installa con: pip install torch scikit-learn")
        sys.exit(1)

    model = train_cnn()

    # Step 3: Test rapido su un simbolo generato al volo
    print("\n[ML] Test rapido su simbolo generato:")
    model.eval()
    device = next(model.parameters()).device
    for sym in SYMBOLS:
        vec = _render_symbol(sym)   # rendering canonico, senza augmentazione
        arr = vec.reshape(CELL_SIZE, CELL_SIZE)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_idx = model(tensor).argmax(dim=1).item()
        pred = IDX_TO_SYMBOL[pred_idx]
        ok   = "OK" if pred == sym else f"ERRORE (predetto: {pred})"
        print(f"  Simbolo reale: {sym} -> Predetto: {pred} [{ok}]")
