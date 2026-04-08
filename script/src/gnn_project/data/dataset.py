import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 피처 레이아웃 (통일 벡터, 크기 = 11)
# ─────────────────────────────────────────────
# 인덱스  피처         upstream(1~28)  outlet(29)
#   0    FLOW              ✅              ✅
#   1    TN                ✅              ✅
#   2    TP                ✅              ✅
#   3    PCP               0 (패딩)        ✅
#   4    wt                0 (패딩)        ✅
#   5    chl               0 (패딩)        ✅  ← 과거값(입력윈도우), 예측대상
#   6    Green             0 (패딩)        ✅
#   7    Red               0 (패딩)        ✅
#   8    Blue_Green        0 (패딩)        ✅
#   9    Red_NIR           0 (패딩)        ✅
#  10    Node Type         0 (Upstream)   1 (Outlet)
# ─────────────────────────────────────────────
# Target: outlet 노드의 chl (t+1 시점), 스케일링된 값
FEATURE_DIM = 11   # 통일 피처 벡터 크기
RAW_DIM     = 10   # 패딩 전 피처 수 (Node Type 제외)
OUTLET_IDX  = 28   # 0-based (sub=29)
N_NODES     = 29


def load_real_data(outlet_path: str, upstream_path: str):
    """
    outlet_lag_pcp.csv + upstream_lag_pcp.csv를 읽어
    raw_features [T, N, RAW_DIM] 배열로 조립합니다.

    - lag 컬럼은 사용하지 않습니다 (현재값만 사용).
    - outlet에서: FLOW, TN, TP, PCP, wt, chl, Green, Red, Blue_Green, Red_NIR
    - upstream에서: FLOW, TN, TP  (나머지 인덱스 3~9 = 0)

    Returns:
        raw   : np.ndarray [T, N, RAW_DIM], dtype=float32
        dates : list of pd.Timestamp (관측 날짜 순서)
    """
    outlet_df   = pd.read_csv(outlet_path,   parse_dates=['date'])
    upstream_df = pd.read_csv(upstream_path, parse_dates=['date'])

    # outlet 날짜 기준으로 정렬
    dates = sorted(outlet_df['date'].unique())
    T = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    raw = np.zeros((T, N_NODES, RAW_DIM), dtype=np.float32)

    # ── Outlet 노드 (index 28) ──────────────────────────────
    outlet_cols = ['FLOW', 'TN', 'TP', 'PCP', 'wt', 'chl',
                   'Green', 'Red', 'Blue_Green', 'Red_NIR']
    out_sub = outlet_df[outlet_df['sub'] == 29].copy()
    out_sub['t_idx'] = out_sub['date'].map(date_to_idx)
    out_sub = out_sub.dropna(subset=['t_idx'])
    out_sub['t_idx'] = out_sub['t_idx'].astype(int)

    t_idxs = out_sub['t_idx'].values
    for f_i, col in enumerate(outlet_cols):
        raw[t_idxs, OUTLET_IDX, f_i] = out_sub[col].values

    # ── Upstream 노드 (sub 1~28, indices 0~27) ──────────────
    up_sub = upstream_df[upstream_df['sub'] != 29].copy()
    up_sub['t_idx'] = up_sub['date'].map(date_to_idx)
    up_sub = up_sub.dropna(subset=['t_idx'])
    up_sub['t_idx'] = up_sub['t_idx'].astype(int)
    up_sub['n_idx'] = up_sub['sub'].astype(int) - 1  # sub 1 → index 0

    t_idxs = up_sub['t_idx'].values
    n_idxs = up_sub['n_idx'].values
    raw[t_idxs, n_idxs, 0] = up_sub['FLOW'].values
    raw[t_idxs, n_idxs, 1] = up_sub['TN'].values
    raw[t_idxs, n_idxs, 2] = up_sub['TP'].values
    # 인덱스 3~9는 0 유지 (upstream에 해당 피처 없음)

    return raw, dates


# ─────────────────────────────────────────────
# Dataset 클래스
# ─────────────────────────────────────────────
class OutletPredictionDataset(Dataset):
    """
    미호천 단일 유출구 예측을 위한 시공간 시계열 데이터셋.
    Data Leakage(t시점 예측 시 t시점 입력 사용)를 원천 차단합니다.
    """
    def __init__(self, node_features, outlet_target, lookback_window=14, name="Dataset"):
        """
        :param node_features:  [T, N, FEATURE_DIM] (스케일링 + 패딩 완료)
        :param outlet_target:  [T, 1]              (스케일링된 chl 정답)
        :param lookback_window: 관측 윈도우 크기 k
        """
        super().__init__()
        self.node_features    = torch.as_tensor(node_features, dtype=torch.float32)
        self.targets          = torch.as_tensor(outlet_target, dtype=torch.float32)
        self.lookback_window  = lookback_window
        self.name             = name

        self.T, self.N, self.F = self.node_features.shape
        self.num_samples = self.T - self.lookback_window

        if self.num_samples <= 0:
            raise ValueError(
                f"[{name}] 데이터 길이({self.T})가 윈도우 크기({self.lookback_window})보다 작거나 같습니다."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 입력: [idx, idx+L) → shape [L, N, F]
        x_seq = self.node_features[idx : idx + self.lookback_window]
        # 정답: 윈도우 바로 다음 시점
        target_idx = idx + self.lookback_window
        y_target   = self.targets[target_idx]
        return x_seq, y_target, target_idx


# ─────────────────────────────────────────────
# 데이터 분할 + 스케일링 + 패딩
# ─────────────────────────────────────────────
def prepare_and_split_data(raw_features, outlet_node_idx=OUTLET_IDX,
                            lookback_window=14, train_ratio=0.7, val_ratio=0.15,
                            apply_log1p=True):
    """
    엄격한 Chronological Split, 피처별 스케일링, 통일 패딩 적용.

    :param raw_features: [T, N, RAW_DIM]
                         upstream: [:3] = FLOW, TN, TP  / [3:] = 0
                         outlet  : [:10] = 전체 피처
    :param apply_log1p: 비대칭 분포를 가진 피처(FLOW, TN, TP, chl)에 np.log1p 사전 적용
    Returns: train_ds, val_ds, test_ds, scaler_out, scaler_target
    """
    T, N, F_raw = raw_features.shape
    assert F_raw == RAW_DIM, f"raw_features 마지막 차원이 {RAW_DIM}이어야 합니다. 현재: {F_raw}"

    upstream_indices = [i for i in range(N) if i != outlet_node_idx]

    # 1. Log Transform (선택)
    # 인덱스: FLOW(0), TN(1), TP(2), chl(5)
    features_log = raw_features.copy()
    if apply_log1p:
        skewed_up  = [0, 1, 2]
        skewed_out = [0, 1, 2, 5]
        
        # 음수값 예외처리를 위한 clip
        up_subset = features_log[:, upstream_indices, :].copy()
        up_subset[:, :, skewed_up] = np.log1p(np.clip(up_subset[:, :, skewed_up], 0, None))
        features_log[:, upstream_indices, :] = up_subset
        
        out_subset = features_log[:, outlet_node_idx, :].copy()
        out_subset[:, skewed_out] = np.log1p(np.clip(out_subset[:, skewed_out], 0, None))
        features_log[:, outlet_node_idx, :] = out_subset

    # 2. Chronological Split
    train_end = int(T * train_ratio)
    val_end   = train_end + int(T * val_ratio)

    train_times = np.arange(0, train_end)
    val_times   = np.arange(train_end, val_end)
    test_times  = np.arange(val_end, T)
    assert train_times[-1] < val_times[0] < test_times[0], "Temporal Split 누수 발생"

    X_train = features_log[:train_end].copy()
    X_val   = features_log[train_end:val_end].copy()
    X_test  = features_log[val_end:].copy()

    # 3. Train 기준으로 Scaler 학습
    def fit_scalers(train_arr):
        # Upstream: FLOW, TN, TP (채널 0~2)
        scaler_up = StandardScaler()
        up_data = train_arr[:, upstream_indices, :3].reshape(-1, 3)
        scaler_up.fit(up_data)

        # Outlet: 10채널 전체
        scaler_out = StandardScaler()
        out_data = train_arr[:, outlet_node_idx, :RAW_DIM].reshape(-1, RAW_DIM)
        scaler_out.fit(out_data)
        
        # Target: 오직 Chl-a (역변환을 쉽고 확실하게 하기 위함)
        scaler_target = StandardScaler()
        target_data = train_arr[:, outlet_node_idx, 5].reshape(-1, 1)
        scaler_target.fit(target_data)

        return scaler_up, scaler_out, scaler_target

    scaler_up, scaler_out, scaler_target = fit_scalers(X_train)

    # 4. 스케일링 + 통일 패딩([T, N, FEATURE_DIM=11])
    def transform_and_pad(arr):
        out_arr = np.zeros((arr.shape[0], N, FEATURE_DIM), dtype=np.float32)

        # Upstream 변환 (채널 0~2만 채움, 나머지 0)
        up_orig   = arr[:, upstream_indices, :3]
        up_scaled = scaler_up.transform(up_orig.reshape(-1, 3)).reshape(up_orig.shape)
        out_arr[:, upstream_indices, :3] = up_scaled
        out_arr[:, upstream_indices, 10] = 0.0   # Node Type = Upstream

        # Outlet 변환 (채널 0~9 전부 채움)
        out_orig   = arr[:, outlet_node_idx, :RAW_DIM]
        out_scaled = scaler_out.transform(out_orig.reshape(-1, RAW_DIM)).reshape(out_orig.shape)
        out_arr[:, outlet_node_idx, :RAW_DIM] = out_scaled
        out_arr[:, outlet_node_idx, 10]       = 1.0  # Node Type = Outlet

        # 정답 y: chl = (target scaler 기준 변환 적용)
        y_orig = arr[:, outlet_node_idx, 5].reshape(-1, 1)
        y_scaled = scaler_target.transform(y_orig)

        # 주의: 피처로서 사용될 때는 scaler_out에 의해 변환된 out_scaled[:, 5] 값을 유지함
        # 하지만 명시적으로 분리된 y값은 scaler_target을 사용.
        return out_arr, y_scaled

    X_tr, y_tr = transform_and_pad(X_train)
    X_v,  y_v  = transform_and_pad(X_val)
    X_te, y_te = transform_and_pad(X_test)

    # 5. Padding Sanity Check
    assert np.all(X_tr[:, upstream_indices, 3:10] == 0), \
        "Upstream 패딩 실패: 채널 3~9에 0이 아닌 값 존재"
    assert np.all(X_tr[:, outlet_node_idx, 10] == 1.0), \
        "Outlet Node Type 패딩 실패"

    train_ds = OutletPredictionDataset(X_tr, y_tr, lookback_window, name="Train")
    val_ds   = OutletPredictionDataset(X_v,  y_v,  lookback_window, name="Val")
    test_ds  = OutletPredictionDataset(X_te, y_te, lookback_window, name="Test")

    return train_ds, val_ds, test_ds, scaler_out, scaler_target


# ─────────────────────────────────────────────
# 더미 데이터 생성 (파이프라인 단위 테스트용)
# ─────────────────────────────────────────────
def generate_custom_dummy(T=200, N=N_NODES, F=RAW_DIM):
    """
    실제 데이터와 동일한 형상의 가상 시계열을 생성합니다.
    upstream 노드의 채널 3~9는 0으로 유지합니다.
    """
    raw = np.zeros((T, N, F), dtype=np.float32)
    upstream_indices = [i for i in range(N) if i != OUTLET_IDX]

    # upstream: 채널 0~2만 무작위
    raw[:, upstream_indices, :3] = np.random.randn(T, len(upstream_indices), 3).astype(np.float32)
    # outlet: 채널 0~9 전부 무작위
    raw[:, OUTLET_IDX, :] = np.random.randn(T, F).astype(np.float32)

    return raw


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== [Data Pipeline Test] ===")
    raw = generate_custom_dummy(T=200)
    train_ds, val_ds, test_ds, scaler, scaler_target = prepare_and_split_data(raw, lookback_window=10)

    print(f"Train / Val / Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    bx, by, bidx = next(iter(loader))
    print(f"Batch X shape : {bx.shape}  (기대: [4, 10, 29, 11])")
    print(f"Batch y shape : {by.shape}  (기대: [4, 1])")
    print("✅ 파이프라인 정상 동작")
