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

## 🛠️ Installation & Setup (For Team Members)

1. **Clone the repository**
```bash
git clone https://github.com/kona0107/EGU26-SWAT-GNN.git
cd EGU26-SWAT-GNN
```

2. **Create and Activate a Virtual Environment**
It is highly recommended to isolate dependencies using a virtual environment (`venv`).
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install Dependencies**
Ensure you have the virtual environment activated, then install the required packages.
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install torch_geometric
```

## 📝 Usage (Testing the Pipeline)

We have created a fully integrated testing script to verify the data ingestion and model forward pass using the sample SWAT dataset.

**Run the Integration Test:**
```bash
python script/src/gnn_project/test_run.py
```
*Expected Output:*
The script will output the structural shapes of the data tensors entering the Spatio-Temporal Graph Neural Network (ST-GNN) and confirm that the final predictive outputs match exactly `[Batch, Nodes, OutFeatures(1)]`.

## 📜 License
[MIT License](LICENSE) (Change to Apache 2.0 or appropriate license based on your preference)
