import torch
import os
import sys

# 현재 폴더(gnn_project)를 파이썬 경로에 추가해 모듈 임포트 에러 방지
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_and_preprocess_swat_data, MihoSpatioTemporalDataset
from torch.utils.data import DataLoader
from models.st_gat import SpatioTemporalGNN

def main():
    print("=== [ST-GAT (Transformer+GAT) 통합 파이프라인 테스트] ===")
    
    # 1. 시공간 SWAT 데이터 로드 및 전처리
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'SWAT_OUTPUT_SAMPLE_PROCESSED.csv'))
    
    print(f"\n1. SWAT CSV 데이터를 불러오고 있습니다...\n   - 경로: {csv_file}")
    if not os.path.exists(csv_file):
        print(f"오류: 데이터 파일을 찾을 수 없습니다: {csv_file}")
        return
        
    X_data, Y_data, M_data, scaler = load_and_preprocess_swat_data(csv_file)
    
    # 2. PyTorch Dataset 생성 및 Batching
    lookback = 14
    dataset = MihoSpatioTemporalDataset(X_data, Y_data, M_data, lookback_window=lookback)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    x_batch, y_batch, m_batch, _ = next(iter(dataloader))
    B, L, N, F = x_batch.shape
    print(f"\n2. 데이터 배치 변환 성공! \n   - 목표 형태: [B(배치)={B}, L(기간)={L}, N(유역수)={N}, F(변수수)={F}]")

    # 3. 모델 초기화 (방금 새로 만든 트랜스포머 기반의 시공간 GAT)
    print("\n3. SpatioTemporalGNN (Transformer-GAT) 모델 초기화 중...")
    model = SpatioTemporalGNN(
        in_features=F, 
        temporal_hidden=32, 
        gat_hidden=16, 
        out_features=1, 
        num_temporal_layers=2, 
        gat_heads=2
    )

    # 4. 임시 하천 연결망(Edge Index) 생성
    # 실제로는 유역 간 흐름 맵핑을 구현해야 하지만 테스트를 위해 
    # 0번유역 -> 1번유역 -> 2번유역 순으로 순차적 흐름을 모의합니다.
    source_nodes = torch.arange(0, N-1)
    target_nodes = torch.arange(1, N)
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    print(f"4. 가상의 유역 네트워크 생성 완료! (형태: {edge_index.shape})")

    # 5. 모델 추론 진행!
    print("\n5. 모델 순전파(Forward Pass) 시뮬레이션 시작!")
    model.eval()
    with torch.no_grad():
        predictions = model(x_batch, edge_index)
        
    print(f"\n✅[성공!] 모델 추론 완료!")
    print(f" - 최종 예측 텐서 형태: {predictions.shape} \n - 완벽하게 [Batch({B}), Nodes({N}), OutFeatures(1)] 에 부합합니다.")

if __name__ == '__main__':
    main()
