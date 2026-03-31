# HydroGAT (SWAT-STGAT)

A Spatio-Temporal Graph Attention Network (ST-GAT) framework for modeling hydrological processes using SWAT data. 

**Note: This project is part of the research presented at EGU26.**

## 📌 Project Overview
This repository contains the official PyTorch implementation of a Spatio-Temporal Graph Neural Network (ST-GAT) designed to process SWAT (Soil and Water Assessment Tool) dataset for the Miho River Basin. The model captures both complex topological routing (spatial dependencies) and time-series hydrological patterns (temporal dependencies).

## 🚀 Key Features
- **Spatial Modeling**: Utilizes Graph Attention Networks (GAT) to dynamically weigh the importance of connected sub-basins.
- **Temporal Modeling**: Integrates temporal layers to process sequential time-series SWAT records.
- **Custom Dataset Loading**: Includes `MihoSpatioTemporalDataset` for structured batch processing of spatial routing and temporal variables.
- **EGU26**: Structured for academic research and reproducible experiments.

## 📁 Repository Structure
```
├── script/
│   ├── src/
│   │   ├── gnn_project/
│   │   │   ├── models/        # ST-GAT, GAT, Temporal architectures
│   │   │   ├── losses.py      # Custom loss functions
│   │   │   └── ...
│   ├── data/                  # SWAT data, adjacency matrices (Exclude from Git)
│   ├── configs/               # Hyperparameter configurations
└── document/                  # Supporting academic documents
```

## 🛠️ Installation & Setup (WIP)
1. Clone this repository
```bash
git clone https://github.com/username/hydrogat.git
cd hydrogat/script
```

2. Install dependencies (Requires PyTorch, PyTorch Geometric)
```bash
pip install -r requirements.txt # (To be added)
```

## 📝 Usage
(Details to be provided: How to train the model, start evaluations, etc.)

## 📜 License
[MIT License](LICENSE) (Change to Apache 2.0 or appropriate license based on your preference)
