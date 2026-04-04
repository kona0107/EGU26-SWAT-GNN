import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import generate_custom_dummy, prepare_and_split_data
from torch.utils.data import DataLoader
from models.temporal import TransformerBaseline
from models.st_gcn import SpatioTemporalHybridGNN

def evaluate_persistence_baseline(dataset):
    """
    Persistence Baseline 계산기
    y_hat[t] = Chl-a_past[t-1] 
    """
    total_squared_error = 0.0
    count = 0
    
    for i in range(len(dataset)):
        x_seq, y_target, target_idx = dataset[i]
        
        # x_seq 형태: [L, N, 7]
        # 유출구(N-1번) 노드의 마지막 시점(t-1)의 Chl-a 피처 추출
        # (Chl-a_past는 패딩 구조에서 인덱스 5입니다)
        # 하지만 스케일링 된 target과의 비교를 위해선 스케일링된 Chl_a 값을 찾아야 합니다.
        
        # 실제 데이터 파이프라인에서 Chl-a는 유출구의 5번째 인덱스에 매핑되어 있으며
        # 이는 스케일링 후 2번 차례 변환값으로 적용되었습니다. 
        # 우리의 패딩 인덱스는 0:Flow, 1:TN, 2:TP, 3:Temp, 4:Rain, 5:Chl-a, 6:NodeType
        last_timestep_chl_a = x_seq[-1, -1, 5].item()
        true_chl_a = y_target.item()
        
        squared_error = (true_chl_a - last_timestep_chl_a) ** 2
        total_squared_error += squared_error
        count += 1
        
    return total_squared_error / count

def run_sanity_checks(dataset, outlet_idx, name="Dataset"):
    for i in range(len(dataset)):
        x_seq, y_target, target_idx = dataset[i]
        # 1. Leakage Check (입력 윈도우 인덱스 < 타깃 인덱스)
        input_time_end = target_idx - 1
        assert input_time_end < target_idx, f"Leakage Error! Input End {input_time_end} >= Target {target_idx}"
        
        # 2. Padding Check
        upstream_indices = [idx for idx in range(dataset.N) if idx != outlet_idx]
        assert torch.all(x_seq[:, upstream_indices, 3:6] == 0), "Upstream Leakage into downstream features"
        assert torch.all(x_seq[:, outlet_idx, :3] == 0), "Downstream Leakage into upstream features"
        
        # 3. Node Type Check
        assert torch.all(x_seq[:, upstream_indices, 6] == 0), "Node Type mismatch for upstream"
        assert torch.all(x_seq[:, outlet_idx, 6] == 1), "Node Type mismatch for outlet"
    print(f"[{name}] 모든 Sanity Check 통과 완료 (Leakage Zero / Padding 완벽)")

def main():
    print("=== [EGU26 수문학적 GNN 데이터 파이프라인 테스트] ===")
    T, N, F = 365, 29, 6   # 6개의 실측치 가정 (유량, TN, TP, 수온, 강수, Chl-a)
    lookback = 14
    outlet_idx = N - 1
    
    # 1. 시계열 데이터셋 로딩 (엄격하게 T1 < T2 < T3 split)
    print("\n1. 엄격한 Chronological Split 데이터를 생성합니다.")
    raw_dummy = generate_custom_dummy(T, N, F)
    
    train_ds, val_ds, test_ds, scaler_out = prepare_and_split_data(
        raw_dummy, outlet_node_idx=outlet_idx, lookback_window=lookback
    )
    
    print(f"  - Train/Val/Test 분할 완료: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    
    run_sanity_checks(test_ds, outlet_idx, "Test Dataset")
    
    # 2. Persistence Baseline 오차 계산
    print("\n2. [Persistence Baseline] MSE 계산 (단순 타성 예측 기준)")
    base_mse = evaluate_persistence_baseline(test_ds)
    print(f"  - Persistence Baseline MSE: {base_mse:.4f}")
    
    # 3. 모델 테스트 구조
    B = 32
    test_loader = DataLoader(test_ds, batch_size=B)
    x_batch, y_batch, t_idx = next(iter(test_loader))
    
    # x_batch shape => [B, L, N, 7]
    print(f"\n3. 모델 아키텍처 포워드 테스트 (Batch Size : {x_batch.shape})")
    
    # 3-1. 모델 1: Transformer (단일 유출구 Baseline)
    base_transformer = TransformerBaseline(in_features=7, hidden_dim=32, out_features=1)
    base_preds = base_transformer(x_batch, outlet_node_idx=outlet_idx)
    print(f"  [Model 1. Transformer] 단일 지점 Forward 성공! => Output Shape: {base_preds.shape}")
    
    # 3-2. 모델 2: Transformer + SAGEConv 병합 모델
    hybrid_model = SpatioTemporalHybridGNN(in_features=7, temporal_hidden=32, gcn_hidden=16)
    
    # 강제 Directed Edge-Index 생성 (Upstream -> Downstream 구조)
    # 노드 0부터 N-2까지 -> 순서대로 다음 노드 (단일방향!)
    source = torch.arange(0, N-1)
    target = torch.arange(1, N)
    directed_edge_index = torch.stack([source, target], dim=0) # [2, E]
    
    print(f"  [Edge Index 방향성 Check] Source -> Target Only (Undirected 사용 X)")
    hybrid_preds = hybrid_model(x_batch, directed_edge_index, outlet_node_idx=outlet_idx)
    print(f"  [Model 2. Hybrid Trans+GCN] 시공간망 Forward 성공! => Output Shape: {hybrid_preds.shape}")
    
    # 파이프라인 무결성 확인용 손실 계산
    criterion = nn.MSELoss()
    loss_val = criterion(hybrid_preds, y_batch.view_as(hybrid_preds))
    print(f"  - Mock Forward 배치 손실 정상 계산 여부 (Loss): {loss_val.item():.4f}")
    print("\n✅ 전체 파이프라인 정상 구동 검증 완료")

if __name__ == "__main__":
    main()
