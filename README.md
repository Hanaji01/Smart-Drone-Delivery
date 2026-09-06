# Smart Drone Delivery

Smart Drone Delivery is a Python-based project that simulates an autonomous drone responsible for completing multiple deliveries on a grid-based environment.

The system combines **image processing**, **machine learning**, **deep learning**, and **artificial intelligence search algorithms** to interpret a map, identify its cells, plan a route, and simulate the drone's movement.

The project includes a graphical interface developed with **Pygame** and a convolutional neural network (CNN) implemented with **PyTorch** for recognizing the symbols that define the map.

---

## Project Overview

The drone operates on a two-dimensional grid containing different types of cells:

* `S` — Starting position
* `D` — Delivery point
* `W` — Wind zone
* `X` — Blocked / no-fly zone
* `E` — Empty cell

The objective is to visit all delivery points and then return to the starting position.

The route is planned using artificial intelligence search algorithms. The system supports several algorithms and allows their performance to be compared.

---

## Main Features

* Grid-based drone environment
* Image-based map recognition
* CNN-based cell classification
* Automatic map generation for testing
* Multiple delivery points
* Obstacles and no-fly zones
* Wind zones with increased movement cost
* Autonomous path planning
* Nine implemented search algorithms
* Graphical visualization with Pygame
* Animated drone movement
* Search performance statistics
* Algorithm comparison
* Reproducible map generation using random seeds

---

## Machine Learning Component

The map is provided as an image and divided into individual cells.

Each cell is classified by a convolutional neural network into one of five classes:

```text
S → START
D → DELIVERY
W → WIND
X → BLOCKED
E → EMPTY
```

The CNN works with grayscale images of **28 × 28 pixels**.

### CNN Architecture

The neural network consists of two convolutional blocks followed by fully connected layers:

```text
Input: 1 × 28 × 28

Conv2D: 1 → 16
ReLU
MaxPool

Conv2D: 16 → 32
ReLU
MaxPool

Flatten

Linear: 32 × 7 × 7 → 128
ReLU

Linear: 128 → 5
```

The five output classes correspond to the five possible cell types.

---

## Dataset

The project includes a dataset used to train the CNN.

The dataset contains five classes:

```text
dataset/
├── S/
├── D/
├── W/
├── X/
└── E/
```

The configured training dataset contains **1,000 samples per class**, for a total of **5,000 images**.

Images are generated automatically and include small rotations and Gaussian noise to introduce variation into the training data.

The dataset is divided into:

* **80% training data**
* **20% test data**

The model is trained using the **Adam optimizer** and **Cross-Entropy Loss**.

---

## Image Parsing

The `image_parser.py` module converts a map image into the internal grid representation.

The image is divided into 28 × 28 pixel cells. Each cell is passed to the trained CNN, which predicts its corresponding symbol.

The predicted symbols are then converted into `GridCell` objects and used to construct the complete `Grid`.

For example:

```text
SDEWXEEE
EXXEEXEE
EEXEEEED
EEEWWEEE
EWEWEWEE
EEEEEEEE
EEEDEXEE
EEEEEEEE
```

This representation can then be converted into the internal environment used by the search algorithms.

---

## Environment

The `environment.py` module defines the grid and its cells.

Each cell has one of the following types:

| Symbol | Type     | Description                       |     Cost |
| ------ | -------- | --------------------------------- | -------: |
| `S`    | START    | Drone starting and final position |        1 |
| `D`    | DELIVERY | Delivery destination              |        1 |
| `W`    | WIND     | Adverse wind zone                 |        2 |
| `X`    | BLOCKED  | No-fly zone                       | Infinite |
| `E`    | EMPTY    | Normal traversable cell           |        1 |

Blocked cells cannot be entered.

Wind cells are traversable, but moving through them has double the normal cost.

---

## Path Planning

The `controller.py` module implements nine search algorithms:

1. Breadth-First Search (BFS)
2. Depth-First Search (DFS)
3. Iterative Deepening Search (IDS)
4. Uniform-Cost Search (UCS)
5. Greedy Best-First Search
6. A*
7. IDA*
8. Recursive Best-First Search (RBFS)
9. Simplified Memory-Bounded A* (SMA*)

The default algorithm used by the graphical application is **A***.

---

## A* Search

A* evaluates each state using:

```text
f(n) = g(n) + h(n)
```

where:

* `g(n)` is the accumulated cost from the initial state
* `h(n)` is the estimated cost to reach the goal

The project uses **Manhattan distance** as the heuristic.

The search state contains both:

```text
Drone position
+
Pending deliveries
```

The goal state is reached when:

```text
All deliveries are completed
AND
The drone has returned to the starting position
```

Because wind zones have a higher traversal cost, the path planner considers not only the number of steps but also the energy cost of the route.

---

## Search State

A search state is represented by:

```text
position
pending_deliveries
steps
```

The set of pending deliveries is stored as a `frozenset`, allowing states with the same position and remaining deliveries to be identified efficiently.

Search nodes also store:

* Parent node
* Action leading to the node
* Path cost `g(n)`
* Heuristic cost `h(n)`
* Search depth

This information allows the system to reconstruct the final path after reaching the goal.

---

## Movement System

The `movement.py` module manages the physical movement of the drone within the grid.

The drone can move in the four cardinal directions:

```text
UP
DOWN
LEFT
RIGHT
```

Before executing a movement, the system checks:

* Whether the destination is inside the grid
* Whether the destination is traversable
* The cost of entering the destination cell
* Whether a delivery is completed
* Whether the drone has entered a wind zone

When the drone reaches a pending delivery cell, the delivery is automatically marked as completed.

---

## Sensors

The `sensors.py` module simulates the drone's sensors.

The sensors keep track of:

* Current position
* Number of steps taken
* Wind conditions

The sensor state can also be saved and restored.

This is particularly useful when comparing multiple search algorithms, because each algorithm can start from the same initial configuration.

---

## Graphical Interface

The graphical interface is implemented using **Pygame**.

The application displays:

* The complete grid
* The drone's current position
* Planned route
* Completed route
* Delivery points
* Wind zones
* Blocked cells
* Number of completed deliveries
* Number of expanded nodes
* Execution time
* Total path cost
* Number of steps
* Simulation progress
* Current algorithm

The drone can be animated along the calculated route.

---

## Controls

The main application supports the following keyboard controls:

| Key         | Action                               |
| ----------- | ------------------------------------ |
| `SPACE`     | Pause / resume simulation            |
| `R`         | Reset the current simulation         |
| `1`         | BFS                                  |
| `2`         | DFS                                  |
| `3`         | IDS                                  |
| `4`         | UCS                                  |
| `5`         | Greedy                               |
| `6`         | A*                                   |
| `7`         | IDA*                                 |
| `8`         | RBFS                                 |
| `9`         | SMA*                                 |
| `+` / `-`   | Increase / decrease simulation speed |
| `C`         | Compare enabled algorithms           |
| `M`         | Generate a new random map            |
| `ESC` / `Q` | Exit                                 |

---

## Search Performance

Each search algorithm produces a `SearchResult` containing several performance metrics:

```text
Algorithm
Path
Total cost
Nodes expanded
Maximum frontier size
Execution time
Success/failure status
```

These measurements make it possible to compare the computational behavior of different search strategies.

The system can therefore be used not only as a drone simulation but also as a practical demonstration of classical AI search algorithms.

---

## Random Map Generation

The project can generate random test maps automatically.

A generated map contains:

* One starting cell
* Multiple delivery cells
* Blocked cells
* Wind cells
* Empty cells

The default test configuration uses an **8 × 8 grid**.

Multiple test maps can also be generated to evaluate the image parser and the complete navigation pipeline under different environments.

---

## Project Structure

```text
smartdronedelivery_NEW/
│
├── main.py
├── controller.py
├── environment.py
├── movement.py
├── sensors.py
├── image_parser.py
├── ml_classifier.py
├── requirements.txt
│
├── cnn_model.pt
│
├── dataset/
│   ├── S/
│   ├── D/
│   ├── W/
│   ├── X/
│   └── E/
│
└── test_map*.png
```

### `main.py`

Main entry point of the application.

It initializes the environment, graphical interface, simulation, user controls, and visualization.

### `controller.py`

Contains the search algorithms used for autonomous path planning.

### `environment.py`

Defines the grid, cell types, movement costs, delivery states, and grid-related operations.

### `movement.py`

Implements the drone movement engine and validates individual movements and complete paths.

### `sensors.py`

Simulates the drone's sensors and tracks its position, movement count, and wind conditions.

### `image_parser.py`

Loads map images and converts their cells into the internal grid representation using the CNN classifier.

### `ml_classifier.py`

Contains the CNN architecture, dataset generation, training, evaluation, model loading, and cell prediction functions.

### `cnn_model.pt`

Saved weights of the trained CNN model.

---

## Technologies

The project is implemented in **Python** and uses the following technologies:

* **Python**
* **PyTorch**
* **NumPy**
* **Pillow**
* **scikit-learn**
* **Pygame**

---

## Installation

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd smartdronedelivery_NEW
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

The graphical application can be started with:

```bash
python main.py
```

If no map image is provided, the application automatically generates and uses a test map.

A specific map can also be provided as a command-line argument:

```bash
python main.py path/to/map.png
```

---

## Training the CNN

The machine learning component can be executed independently:

```bash
python ml_classifier.py
```

This component can:

1. Generate the dataset if necessary.
2. Load the training data.
3. Split the data into training and test sets.
4. Train the CNN.
5. Evaluate the model.
6. Save the trained model.
7. Perform a quick prediction test on the five symbols.

---

## Workflow

The complete system follows this pipeline:

```text
Map Image
    |
    v
Image Processing
    |
    v
CNN Classification
    |
    v
Grid Construction
    |
    v
Search Algorithm
    |
    v
Optimal / Planned Path
    |
    v
Movement Engine
    |
    v
Pygame Simulation
```

This architecture connects the machine learning component with the artificial intelligence path-planning component.

---

## Educational Purpose

The project was developed as an application of concepts related to **Artificial Intelligence**, **Machine Learning**, and **search algorithms**.

It demonstrates how different components can be combined to create an autonomous agent capable of:

* Perceiving an environment from an image
* Representing the environment as a state space
* Identifying objectives
* Planning a route
* Considering different movement costs
* Executing the planned route
* Measuring algorithm performance

---

## Author

**Hanaji Tancre'**

Computer Science
University of Perugia

---

## License

This project is released under the MIT License.
