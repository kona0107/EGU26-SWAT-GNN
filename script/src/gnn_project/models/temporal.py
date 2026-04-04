import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    트랜스포머 모델에 시간적 위치(타임스텝) 정보를 주입하기 위한 함수.
    불규칙한 샘플링 간격을 극복하는 데 핵심적인 역할을 합니다.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, SeqLen, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TemporalEncoder(nn.Module):
    """
    각 Subbasin(노드)의 시간적 다이내믹스를 인코딩하는 <Transformer> 모듈.
    과거 L일간의 데이터를 Attention 매커니즘으로 분석하여, 결측치가 있더라도
    데이터 간의 장기적인 의존성과 딜레이(지연) 효과를 정확하게 포착합니다.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, nhead=4, dropout=0.1):
        """
        초기화 파라미터 설명:
        - input_dim: 입력 피처의 개수 (SWAT 데이터의 Flow, TN, TP + NodeType 등 = 7)
        - hidden_dim: 트랜스포머가 내부적으로 계산할 공간의 크기
        - num_layers: 트랜스포머 인코더 블록을 몇 겹으로 쌓을지
        - nhead: Multi-head Attention 장치의 개수.
        """
        super().__init__()
        # 1. 원본 변수들을 트랜스포머 차원(d_model)으로 선형 변환
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 2. 위치 인코딩
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 3. 트랜스포머 인코더 블록 조립
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [B, L, N, F]
        return: [B, N, hidden_dim]
        """
        B, L, N, F = x.shape
        
        # [B, L, N, F] -> [B, N, L, F] -> [B*N, L, F] 
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N, L, F)
        
        # 차원 매핑: F -> hidden_dim
        x_proj = self.input_projection(x_reshaped)
        
        # 위치 인코딩 추가
        x_pos = self.pos_encoder(x_proj)
        
        # 트랜스포머 인코딩 진행 => [B*N, L, hidden_dim]
        out = self.transformer_encoder(x_pos)
        
        # 다음 스텝을 예측하기 위해 직전 시퀀스인 마지막(-1) 스텝의 Attention 결과를 최종 임베딩으로 사용
        last_hidden = out[:, -1, :]  # [B*N, hidden_dim]
        
        # 다시 원래의 배치와 노드 차원으로 복구 => [B, N, hidden_dim]
        node_embeddings = last_hidden.view(B, N, -1)
        
        return node_embeddings

class TransformerBaseline(nn.Module):
    """
    Model 1. Transformer (단일 유출구 예측용 Baseline)
    입력: 유출구 노드의 시계열 (T-k ~ T-1)
    출력: 유출구 예측 Chl-a[t]
    """
    def __init__(self, in_features=7, hidden_dim=32, out_features=1, num_layers=2, nhead=4):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            input_dim=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_features)
        )
        
    def forward(self, x_seq, outlet_node_idx=-1):
        """
        x_seq: [B, L, N, F]
        outlet_node_idx: 유출구 노드 인덱스
        """
        # 유출구 시계열만 추출 => [B, L, 1, F]
        outlet_seq = x_seq[:, :, outlet_node_idx, :].unsqueeze(2)
        
        # Temporal 인코딩 => [B, 1, hidden_dim]
        embeds = self.temporal_encoder(outlet_seq)
        
        # Squeeze => [B, hidden_dim]
        embeds = embeds.squeeze(1)
        
        # 최종 스칼라 값 예측 => [B, 1]
        out = self.predictor(embeds)
        return out
