# HydroGNN (SWAT-ST-GCN)

미호천 유역의 수문 자료를 활용하여 유출구 클로로필-a(Chl-a) 농도를 예측하는 시공간 그래프 신경망(Spatio-Temporal GNN) 프레임워크입니다.

## Project Status

EGU General Assembly 2026 (EGU26) 발표를 위한 연구 코드입니다.
포스터 발표일: 2026년 5월 5일

본 저장소의 구현체는 연구 진행 상황에 따라 지속적으로 업데이트됩니다.

## 📌 연구 개요

SWAT(Soil and Water Assessment Tool) 보정/검증 결과 및 현장 관측 자료를 활용하여 **미호천 유출구의 주단위 Chl-a 농도**를 예측합니다.

**예측 목표**: 유출구 1개 지점의 `Chl-a[t]`

**입력 자료**:
- 상류 노드: `유량(Flow)`, `총질소(TN)`, `총인(TP)` — 주단위 SWAT 출력값
- 유출구 노드: `수온(Water Temp)`, `강수량(Rain)`, `과거 Chl-a[t-k:t-1]` — 현장 관측값

**핵심 설계 원칙**:
- `t` 시점 예측 시 입력은 반드시 `t-1` 시점까지만 사용 (Data Leakage 원천 차단)
- Train 데이터만으로 Scaler를 fit한 뒤 Zero-Padding 적용 (통계 왜곡 방지)
- Train < Val < Test 시간 순서대로 분할 (Random Split 금지)

## 🗂️ 모델 구성

### Model 1. Transformer (Baseline)
유출구 단일 지점의 시계열만 사용하는 기준 모델입니다.

| 항목 | 내용 |
|---|---|
| 입력 | 유출구 `[수온, 강수량, 과거 Chl-a]` 시계열 |
| 시간 모델링 | Transformer Encoder |
| 출력 | `Chl-a[t]` (스칼라) |

### Model 2. Transformer + SAGEConv (Main)
상류 수질·유량 정보를 공간 그래프를 통해 유출구로 전달하는 시공간 하이브리드 모델입니다.

| 항목 | 내용 |
|---|---|
| 입력 | 전체 노드 `7D` 통일 피처 (Zero-Padding + Node-Type Indicator) |
| 시간 모델링 | Transformer Encoder (노드별 독립 적용) |
| 공간 모델링 | SAGEConv (상류→하류 directed edge 전용) |
| 출력 | 유출구 노드 임베딩 → MLP → `Chl-a[t]` |

**노드 피처 구조 (7D)**:
```
공통 순서: [Flow, TN, TP, Temp, Rain, Chl-a_past, NodeType]
상류 노드: [Flow, TN, TP,    0,    0,           0,        0]
유출구:    [   0,  0,  0, Temp, Rain,  Chl-a_past,        1]
```

## 📁 파일 구조

```
├── script/
│   ├── src/
│   │   └── gnn_project/
│   │       ├── models/
│   │       │   ├── temporal.py    # Transformer Encoder + TransformerBaseline (Model 1)
│   │       │   ├── gcn.py         # SAGEConv 기반 SpatialEncoder
│   │       │   └── st_gcn.py      # 시공간 하이브리드 GNN (Model 2)
│   │       ├── data/
│   │       │   └── dataset.py     # 엄격한 시계열 분할 및 데이터셋 클래스
│   │       └── test_run.py        # 전체 파이프라인 통합 테스트
│   └── data/                      # SWAT 출력, 관측 자료 (Git 제외)
├── document/                      # 연구 관련 문서
└── requirements.txt
```

## 🛠️ 환경 세팅 및 설치 가이드 (팀원용)

1. **저장소 클론(Clone) 받아오기**
```bash
git clone https://github.com/kona0107/EGU26-SWAT-GNN.git
cd EGU26-SWAT-GNN
```

2. **파이썬 가상환경 생성 및 실행**
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **필수 라이브러리 설치**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install torch_geometric
```

## 📝 파이프라인 통합 테스트 실행

더미 데이터로 전체 파이프라인(데이터 분할 → Scaling → Padding → 모델 Forward → 검증)이 정상 작동하는지 확인합니다.

```bash
python script/src/gnn_project/test_run.py
```

**정상 실행 시 출력 내용:**
- Train/Val/Test 시간순 분할 완료 메시지
- Padding/Leakage/Node-Type Sanity Check 통과 확인
- Persistence Baseline MSE 출력
- Model 1 (Transformer) Forward 성공 → Output Shape: `[B, 1]`
- Model 2 (Hybrid Trans+GCN) Forward 성공 → Output Shape: `[B, 1]`
- `✅ 전체 파이프라인 정상 구동 검증 완료`

## 🤝 팀 협업 가이드 (Branch & Convention)

### 1. 브랜치(Branch) 작업 흐름

```bash
# 1. 최신 코드 가져오기
git pull origin main

# 2. 본인 작업용 브랜치 생성
git checkout -b feature/기능이름   # 예: feature/add-training-loop

# 3. 코드 수정 후 커밋
git add .
git commit -m "Feat: 학습 루프 추가"

# 4. 브랜치 업로드 후 PR 생성
git push origin feature/기능이름
```

### 2. 커밋 메시지 규칙

- `Feat:` — 새로운 기능 추가
- `Fix:` — 버그 수정
- `Refactor:` — 코드 구조 개선 (동작 변경 없음)
- `Docs:` — README 등 문서 수정
- `Chore:` — 패키지 업데이트, 파일 삭제 등 기타 변경
- `Test:` — 테스트 코드 추가 및 수정

> 용어, 변수명, 클래스명 등 코드 관련 고유명사는 영문 유지, 나머지는 한글 작성

### 3. 코드 컨벤션

- **PEP8** 스타일 준수
- 모든 클래스·함수에 **Docstring** 작성 (파라미터 및 반환값 명시)
- 예시: `temporal.py`, `st_gcn.py` 참고

## 📜 License

[MIT License](LICENSE)
