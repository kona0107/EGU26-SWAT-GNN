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

## 🤝 팀 협업 가이드 (Branch & Convention)

팀원들과의 안전하고 효율적인 협업을 위해 아래의 규칙을 지켜주세요.

### 1. 브랜치(Branch) 생성 및 작업 흐름 (Workflow)
원격 저장소의 `main` 브랜치에 직접 코드를 올리는 것(Direct Push)은 권장하지 않습니다. 항상 본인의 브랜치를 만들어 작업해 주세요.

```bash
# 1. 최신 코드 가져오기
git pull origin main

# 2. 본인의 작업용 새 브랜치 생성 및 이동
git checkout -b feature/기능이름   # 예: feature/add-transformer, fix/data-loader

# 3. 코드 수정 후 변경사항 저장
git add .
git commit -m "feat: 트랜스포머 모델 구조 작성"

# 4. 내 브랜치를 깃허브에 업로드
git push origin feature/기능이름
```
업로드 후, 깃허브 웹사이트에 접속하여 **Pull Request (PR)**를 생성해 팀원들의 리뷰를 받고 `main` 브랜치로 병합(Merge)합니다.

### 2. 커밋 메시지 규칙 (Commit Message Convention)
커밋 메시지는 다른 팀원이 한눈에 알아볼 수 있도록 머릿말을 달아주세요.
- `feat:` : 새로운 기능 추가
- `fix:` : 버그, 에러 수정
- `docs:` : README 등 문서 수정
- `refactor:` : 결과는 같지만 코드 구조를 개선(리팩토링)
- `chore:` : 패키지 업데이트, 주석 수정 등 자잘한 변경

### 3. 코드 컨벤션 (Code Convention)
- 파이썬 표준 가이드인 **PEP8** 스타일을 지향합니다.
- 새로운 함수나 클래스를 작성할 때는 반드시 **Docstring (`""" 설명 """`)**을 달아 파라미터와 반환값을 명시해 주세요. (예: `st_gat.py` 참고)

## 📜 License
[MIT License](LICENSE) (Change to Apache 2.0 or appropriate license based on your preference)
