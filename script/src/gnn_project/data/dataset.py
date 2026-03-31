import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

class MihoSpatioTemporalDataset(Dataset):
    """
    미호천 GNN 모델을 위한 시공간 슬라이딩 윈도우 데이터셋.
    
    출력 형태:
      - X (입력 시퀀스): [L, N, F] (L: lookback window, N: subbasin 개수, F: feature 개수)
      - y (타겟, Chl-a): [N, 1]
      - mask (품질/결측 마스크): [N, 1] 
    """
    def __init__(self, node_features, targets, masks, lookback_window=14, target_offset=1):
        """
        초기화 메서드.
        
        :param node_features: [T, N, F] 형태의 데이터 (T: 전체 날짜 수, N: 29개 유역, F: 기상·수문 변수)
        :param targets: [T, N, 1] 형태의 정답 타겟 (예: 위성 기반 Chl-a)
        :param masks: [T, N, 1] 형태의 마스크 (관측됨=1, 결측/낮은 신뢰도=0)
        :param lookback_window: L일 (과거 타임스텝 길이)
        :param target_offset: 예측할 시점 간격(1이면 마지막 시퀀스의 다음날 예측)
        """
        super().__init__()
        
        # 텐서 변환 (입력이 numpy일 경우)
        self.node_features = torch.as_tensor(node_features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.float32)
        
        self.lookback_window = lookback_window
        self.target_offset = target_offset
        
        # shape 파악
        self.T, self.N, self.F = self.node_features.shape
        
        # 윈도우 길이를 고려한 총 샘플(Batch) 개수 계산
        self.num_samples = self.T - self.lookback_window - self.target_offset + 1
        
        # 데이터가 너무 짧아 샘플을 만들 수 없는 경우 에러 발생
        if self.num_samples <= 0:
            raise ValueError(
                f"데이터 길이(T={self.T})가 너무 짧습니다. "
                f"lookback_window({self.lookback_window}) + target_offset({self.target_offset}) 이상이어야 합니다."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        데이터 샘플 1개 리턴.
        데이터 로더(DataLoader)가 배치(B)로 묶어 [B, L, N, F], [B, N, 1] 형태로 만들어 줍니다.
        """
        start_idx = idx
        end_idx = idx + self.lookback_window
        
        # 시퀀스 입력: [L, N, F]
        x_seq = self.node_features[start_idx:end_idx]
        
        # 예측 타겟 시점 인덱스 파악
        target_idx = end_idx + self.target_offset - 1
        
        # 타겟과 마스크: [N, 1]
        y_target = self.targets[target_idx]
        target_mask = self.masks[target_idx]
        
        # 나중에 예측값 검증을 위해 타겟 시점(target_idx) 메타데이터도 함께 반환
        return x_seq, y_target, target_mask, target_idx


def load_and_preprocess_swat_data(csv_path):
    """
    SWAT CSV 데이터를 읽어서 [T, N, F] 텐서로 변환하고 정규화합니다.
    """
    df = pd.read_csv(csv_path)
    
    # date와 Sub 기준으로 정렬 (빠뜨림 방지)
    df = df.sort_values(by=['date', 'Sub'])
    
    # 사용할 피처
    feature_cols = ['Flow', 'TN', 'TP']
    
    # 스케일링 (TP 값 스케일이 매우 작으므로 필수)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 차원 파악
    T = df['date'].nunique()
    N = df['Sub'].nunique()
    F = len(feature_cols)
    
    # [T*N, F] -> [T, N, F]
    X_data = df[feature_cols].values.reshape((T, N, F))
    
    # 모델의 목적에 맞게 일단 'Flow' 등을 임시 타겟으로 설정하거나
    # 나중에 관측 데이터(Chl-a 등)와 결합할 수 있습니다.
    # 여기서는 Flow(인덱스 0)를 예측 타겟으로 둡니다.
    Y_data = X_data[:, :, 0:1]
    
    # 결측치 마스크 (모두 1)
    M_data = np.ones((T, N, 1))
    
    return X_data, Y_data, M_data, scaler


def generate_dummy_data(T=365, N=29, F=10):
    """
    형태 확인 및 디버깅을 위한 더미 시계열 텐서 생성기.
    """
    # 1. 시계열 피처 데이터 [T, N, F] (표준정규분포)
    X_dummy = np.random.randn(T, N, F)
    
    # 2. 위성 Chl-a 정답 데이터 [T, N, 1] (0~50 범위 난수)
    y_dummy = np.random.rand(T, N, 1) * 50.0  
    
    # 3. 위성 통과 및 품질 마스크 [T, N, 1] (구름이나 revisit 때문에 15%만 유효값이라 가정)
    #    유효값=1, 결측치=0
    mask_dummy = np.random.choice([0, 1], size=(T, N, 1), p=[0.85, 0.15])
    
    return X_dummy, y_dummy, mask_dummy

if __name__ == "__main__":
    # 데이터셋 정상 작동 테스트 코드
    print("=== [데이터 구조 테스트 시작] ===")
    
    L_lookback = 14
    
    # 1. 실제 SWAT 데이터 로드 시도
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, '..', '..', '..', 'data', 'SWAT_OUTPUT_SAMPLE_PROCESSED.csv')
    
    if os.path.exists(csv_file):
        print(f">> SWAT 데이터 기반 테스트 진행")
        X, Y, M, scaler = load_and_preprocess_swat_data(csv_file)
        T_total, N_nodes, F_features = X.shape
    else:
        print(">> 더미 데이터 기반 테스트 진행")
        T_total, N_nodes, F_features = 365, 29, 3
        X, Y, M = generate_dummy_data(T=T_total, N=N_nodes, F=F_features)
    
    # 2. 데이터셋 인스턴스화
    dataset = MihoSpatioTemporalDataset(X, Y, M, lookback_window=L_lookback)
    
    print(f"전체 날짜 수 (T): {T_total}")
    print(f"생성된 학습/검증 데이터 총 샘플 수: len(dataset) = {len(dataset)}")
    
    # 3. Dataloader 테스트
    from torch.utils.data import DataLoader
    
    # batch_size=32 로더 생성 (shuffle=False 권장: 시계열 확인용)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 첫 배치 형태 출력
    x_batch, y_batch, mask_batch, target_idx_batch = next(iter(loader))
    
    print("\n--- 첫 배치 샘플 shape ---")
    print(f"입력(x_batch): {tuple(x_batch.shape)}  # [B, L, N, F]")
    print(f"타겟(y_batch): {tuple(y_batch.shape)}  # [B, N, 1]")
    print(f"마스크(m_batch): {tuple(mask_batch.shape)}  # [B, N, 1]")
    print(f"타겟 시점 인덱스(target_idx_batch): {tuple(target_idx_batch.shape)}  # [B]")
    print("==========================")
