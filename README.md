# Hopfield Networks & Modern Hopfield Networks

This repository implements both Standard and Modern Hopfield Networks (Dense Associative Memories) and explores their capacity limits and dynamics.

## Files

- **`hopfield.py`**: The core library containing:
  - `HopfieldNetwork`: Standard implementation with Hebbian learning.
  - `ModernHopfieldNetwork`: Implementation using the "Modern" continuous energy function (softmax update).
- **`demo_hopfield.py`**: A comprehensive script that runs:
  - A visual "Vacation Memory" recall demo (recovering noisy images of a penguin, yoga pose, and beach).
  - An energy landscape visualization using PCA.
  - A phase transition/capacity experiment for Standard Hopfield Networks.
  - A phase transition/capacity experiment for Modern Hopfield Networks (demonstrating exponential capacity).

## How to Run

Install dependencies:
```bash
pip install numpy matplotlib
```

Run the demo and experiments:
```bash
python demo_hopfield.py
```

This will display the recovery of patterns and generate phase plots showing the retrieval probability as a function of the number of stored memories ($M$) and noise level.
