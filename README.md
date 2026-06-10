# Data Re-uploading for a Universal Quantum Classifier — Qiskit Reproduction

A reproduction study of the paper **"Data Re-uploading for a Universal Quantum Classifier"** by Pérez-Salinas et al. (2020), implemented in Python and Qiskit for the Laboratory of Computational Physics Project.

> Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020).  
> *Data re-uploading for a universal quantum classifier.*  
> **Quantum, 4**, 226. https://doi.org/10.22331/q-2020-02-06-226

---

## 👥 Group Members

- **Melissa Nespeque**
- **Elena Niero**
- **Eylul Cagla**
- **Sahasra Sivakumar**

---

## 🎯 Project Objective

This project reproduces and analyzes the quantum classification framework proposed in the original paper, which introduces the concept of **data re-uploading**: a technique that embeds classical input data multiple times throughout a variational quantum circuit, enabling a single qubit to act as a universal classifier.

The key idea is that a qubit's state is manipulated by successive single-qubit unitary operations of the form:

$$U(\vec{x}, \vec{\theta}) = U_L(\vec{x}, \vec{\theta}_L) \cdots U_2(\vec{x}, \vec{\theta}_2)\, U_1(\vec{x}, \vec{\theta}_1)$$

where each layer re-encodes the input data $\vec{x}$ alongside trainable parameters $\vec{\theta}_l$, allowing the circuit to learn arbitrarily complex decision boundaries.

The goals of this reproduction are to:
- Implement single-qubit and multi-qubit classifiers using Qiskit
- Train them on synthetic 2D binary classification datasets
- Evaluate accuracy as a function of the number of layers and qubits
- Compare results with those reported in the original paper, including the effect of ancilla qubits

---

## 📁 Repository Structure

├── datasets/
├── results/
│   └── Data_classifier_results.xlsx
├── dataset_generation.py
├── single_qubit_classifier.py
├── multi_qubit_classifier.py
├── single_qubit_experiments.ipynb
├── multi_qubit_experiments.ipynb
├── results_analysis.ipynb
└── README.md

---

## 📓 Notebooks

### `single_qubit_experiments.ipynb`

Implements and trains the **single-qubit variational classifier** across three synthetic datasets and multiple circuit depths.

- Datasets: `circle`, `diamond`, `wavy_lines`
- Layer counts tested: `[1, 2, 4, 6, 8, 12]`
- Circuit: a repeated block of $R_z(\theta_1) R_y(\theta_2) R_z(\theta_3)$ gates interleaved with re-uploaded data encodings
- Each data point $\vec{x} \in \mathbb{R}^3$ is zero-padded to match the 3-parameter encoding per layer
- Optimization: `scipy.optimize.minimize` with the L-BFGS-B method
- Loss function: weighted fidelity cost, measuring overlap with class target states on the Bloch sphere
- Training and test accuracies are tracked per layer count and stored in per-dataset dictionaries

### `multi_qubit_experiments.ipynb`

Extends the classifier to **2- and 3-qubit** circuits, testing both standard measurement and **ancilla-based** measurement schemes.

- Qubit counts: 1, 2, 3
- Measurement types: *without ancilla* (direct fidelity) and *with ancilla* (using an auxiliary qubit to improve class separability)
- Same datasets and layer sweep as the single-qubit case
- Results reveal how expressibility scales with the number of qubits and layers, and the role of the ancilla qubit in improving classification
- Missing configurations (e.g., `wavy_lines / 3 qubits / with ancilla` at layer 8) are recorded as `None` and filtered during analysis

### `results_analysis.ipynb`

Loads the experimental results from `Data_classifier_results.xlsx` and produces comparative visualizations.

- Results are structured as `results[dataset][n_qubits][measurement_type]`, with `layers`, `acc_train`, and `acc_test` lists
- One figure per dataset, with side-by-side train/test accuracy panels
- Color encodes qubit count; line style encodes measurement type (with/without ancilla)
- Highlights convergence behavior and the trade-off between circuit depth and generalization

---

## 🐍 Python Files

### `dataset_generation.py`

Generates the three synthetic 2D classification datasets used throughout the project, following the original paper's GitHub repository.

- `circle`: points inside/outside a circle centered at the origin
- `diamond`: points inside/outside a diamond (rotated square) shape
- `wavy_lines`: points separated by a sinusoidal boundary

Each data point is represented as $\vec{x} = (x_1, x_2, 0)$, where the third dimension is zero-padded to match the 3-component encoding expected by the quantum circuits.  
Stratified train/test splits are created via `sklearn.model_selection.train_test_split`.

### `single_qubit_classifier.py`

Core implementation of the single-qubit classifier pipeline:

- `pack_params(weights, data)` — combines trainable parameters and re-uploaded data into a single parameter vector per layer
- `single_qubit_circuit(params, n_layers)` — builds the Qiskit `QuantumCircuit` with alternating $R_z R_y R_z$ rotations and data re-uploading
- `single_cost_function(params, X, y, n_layers)` — evaluates the weighted fidelity loss over the training set
- `single_qubit_predict(params, X, n_layers)` — returns predicted class labels for a dataset
- Training loop using `scipy.optimize.minimize` (L-BFGS-B) with a callback to record loss per iteration

### `multi_qubit_classifier.py`

Extends the classifier architecture to multiple qubits with optional ancilla measurement:

- Implements entangling layers between qubits using CNOT gates after each re-uploading block
- Supports an ancilla qubit appended to the register, whose measurement is used to determine the class label
- Modular design allowing the number of qubits and measurement strategy to be passed as arguments
- Reuses the same optimization and prediction interface as the single-qubit case

---

## ⚙️ Requirements

qiskit >= 1.0
qiskit-aer
numpy
scipy
scikit-learn
matplotlib
openpyxl

---

## 📊 Results Summary

Training and test accuracies were collected across all combinations of dataset, qubit count, and measurement type, sweeping layer counts `[1, 2, 4, 6, 8, 12]`. Results are stored in `results/Data_classifier_results.xlsx`.

Key findings:
- Accuracy generally improves with depth, with diminishing returns beyond 8 layers
- The `circle` dataset is easiest to classify; `wavy_lines` is the most demanding
- Multi-qubit circuits with ancilla measurement improve classification on complex boundaries
- Results are broadly consistent with those reported in the original paper

---

## 📄 Reference

```bibtex
@article{Perez-Salinas2020,
  title   = {Data re-uploading for a universal quantum classifier},
  author  = {Pérez-Salinas, Adrián and Cervera-Lierta, Alba and Gil-Fuster, Elies and Latorre, José I.},
  journal = {Quantum},
  volume  = {4},
  pages   = {226},
  year    = {2020},
  doi     = {10.22331/q-2020-02-06-226}
}
```
