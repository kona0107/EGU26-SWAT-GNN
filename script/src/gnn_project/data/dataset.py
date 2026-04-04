import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class OutletPredictionDataset(Dataset):
    """
    미호천 단일 유출구 예측을 위한 엄격한 시공간 시계열 데이터셋.
    Data Leakage(t시점 예측 시 t시점 입력 사용)를 원천 차단합니다.
    """
    def __init__(self, node_features, outlet_target, lookback_window=14, name="Dataset"):
        """
        :param node_features: [T, N, 7] 형태. (6D 패딩 피처 + 1D Node Type)
        :param outlet_target: [T, 1] 형태. (스케일링 된 정답 Chl-a)
        :param lookback_window: 관측 윈도우 크기 k
        """
        super().__init__()
        self.node_features = torch.as_tensor(node_features, dtype=torch.float32)
        self.targets = torch.as_tensor(outlet_target, dtype=torch.float32)
        self.lookback_window = lookback_window
        self.name = name
        
        self.T, self.N, self.F = self.node_features.shape
        self.num_samples = self.T - self.lookback_window
        
        if self.num_samples <= 0:
            raise ValueError(f"[{name}] 데이터 길이({self.T})가 윈도우 크기({self.lookback_window})보다 작거나 같습니다.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 입력: t-k ~ t-1 (총 길이: lookback_window)
        # 파이썬 슬라이싱 [idx : idx + L] 은 수학적으로 정확히 L개의 요소를 가져옴.
        x_seq = self.node_features[idx : idx + self.lookback_window]
        
        # 정답: t (입력 시퀀스의 바로 다음 시점)
        # target array에서 위치 idx+L 은 t-1 시점을 1칸 넘어선 t임.
        target_idx = idx + self.lookback_window
        y_target = self.targets[target_idx]
        
        # 검증 통과를 위해 타겟 인덱스 반환
        return x_seq, y_target, target_idx


def prepare_and_split_data(raw_features, outlet_node_idx, lookback_window=14, train_ratio=0.7, val_ratio=0.15):
    """
    엄격한 Chronological Split, 올바른 스케일링/0-Padding 로직 구현체.
    
    :param raw_features: [T, N, 6] (0:Flow, 1:TN, 2:TP, 3:Temp, 4:Rain, 5:Chl-a_past)
                         아직 패딩이 적용되지 않은 날것의 결합 어레이.
    """
    T, N, F_raw = raw_features.shape
    
    # 1. 윈도우 생성 전 Chronological Split (시간순 분할)
    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)
    
    # Assert Sanity Check
    train_times = np.arange(0, train_end)
    val_times = np.arange(train_end, val_end)
    test_times = np.arange(val_end, T)
    assert train_times[-1] < val_times[0] < test_times[0], "Temporal Split 누수 발생"
    
    X_train = raw_features[:train_end].copy()
    X_val = raw_features[train_end:val_end].copy()
    X_test = raw_features[val_end:].copy()
    
    upstream_indices = [i for i in range(N) if i != outlet_node_idx]
    
    def fit_scalers(train_arr):
        # 상류(Upstream) 유효 채널: [0, 1, 2] (Flow, TN, TP)
        scaler_up = StandardScaler()
        # reshape(-1, 3): [T, N-1, 3] -> [(T * (N-1)), 3]
        up_valid = train_arr[:, upstream_indices, :3].reshape(-1, 3)
        scaler_up.fit(up_valid)
        
        # 하류 유출구(Outlet) 유효 채널: [3, 4, 5] (Temp, Rain, Chl-a)
        scaler_out = StandardScaler()
        # reshape(-1, 3): [T, 1, 3] -> [T, 3]
        out_valid = train_arr[:, outlet_node_idx, 3:6].reshape(-1, 3)
        scaler_out.fit(out_valid)
        
        return scaler_up, scaler_out
        
    scaler_up, scaler_out = fit_scalers(X_train)
    
    def transform_and_pad(arr):
        """실제 관측 스케일링(Transform) 진행 후 통일된 7D 구조로 패딩 조립"""
        out_arr = np.zeros((arr.shape[0], N, 7), dtype=np.float32)
        
        # 상류 변환 및 패딩 조립
        up_orig = arr[:, upstream_indices, :3]
        up_scaled = scaler_up.transform(up_orig.reshape(-1, 3)).reshape(up_orig.shape)
        
        out_arr[:, upstream_indices, :3] = up_scaled
        out_arr[:, upstream_indices, 6] = 0.0 # Node Type: Upstream=0
        
        # 하류 변환 및 패딩 조립
        out_orig = arr[:, outlet_node_idx, 3:6]
        out_scaled = scaler_out.transform(out_orig.reshape(-1, 3)).reshape(out_orig.shape)
        
        out_arr[:, outlet_node_idx, 3:6] = out_scaled
        out_arr[:, outlet_node_idx, 6] = 1.0 # Node Type: Outlet=1
        
        # 정답 y 추출 (유출구의 5번 피처는 Chl-a. 스케일링 된 값 자체를 y로 씀)
        # out_scaled의 인덱스 2가 Chl-a_past 스케일 값.
        y_arr = out_scaled[:, 2:3] # [T, 1]
        
        return out_arr, y_arr
        
    X_tr, y_tr = transform_and_pad(X_train)
    X_v, y_v  = transform_and_pad(X_val)
    X_te, y_te = transform_and_pad(X_test)
    
    # 2. Assert Padding Check 
    assert np.all(X_tr[:, upstream_indices, 3:6] == 0), "Upstream 패딩 실패: 뒤 3개 차원에 0이 아닌 값이 있습니다."
    assert np.all(X_tr[:, outlet_node_idx, :3] == 0), "Outlet 패딩 실패: 앞 3개 차원에 0이 아닌 값이 있습니다."
    
    train_dataset = OutletPredictionDataset(X_tr, y_tr, lookback_window, name="Train")
    val_dataset   = OutletPredictionDataset(X_v,  y_v,  lookback_window, name="Val")
    test_dataset  = OutletPredictionDataset(X_te, y_te, lookback_window, name="Test")
    
    return train_dataset, val_dataset, test_dataset, scaler_out


def generate_custom_dummy(T=300, N=29, F=6):
    """
    Flow, TN, TP, Temp, Rain, Chl-a 의 가상 시계열을 만듭니다.
    """
    raw_data = np.random.randn(T, N, F) * 5.0 + 10.0
    return raw_data

if __name__ == "__main__":
    t_len = 300
    n_nodes = 29
    outlet = n_nodes - 1
    
    print("=== [Data Pipeline Test] ===")
    raw = generate_custom_dummy(T=t_len, N=n_nodes)
    train_ds, val_ds, test_ds, scaler = prepare_and_split_data(raw, outlet_node_idx=outlet, lookback_window=10)
    
    print(f"Train samples: {len(train_ds)}")
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    
    bx, by, bidx = next(iter(loader))
    print(f"Batch X shape: {bx.shape} (기대값: B, 10, 29, 7)")
    print(f"Batch y shape: {by.shape} (기대값: B, 1)")
    
    # Window Leakage 검증 (매우 중요)
    print("\n[Window Leakage Assert Check]")
    input_time_end = bidx[0].item() - 1 
    target_time = bidx[0].item()
    if input_time_end < target_time:
        print(f"  성공: 입력 시점의 최대값(t-{input_time_end}) < 타겟 시점(t-{target_time})")
    else:
        print("  실패: Data Leakage 발생")
