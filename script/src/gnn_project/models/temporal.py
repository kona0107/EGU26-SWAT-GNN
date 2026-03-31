import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """
    각 Subbasin(노드)의 시간적 다이내믹스를 인코딩하는 <GRU> 모듈.
    Temporal-first 구조에서 가장 먼저 L일간의 누적된 변화를 학습합니다.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        # batch_first=True 설정: 입력 형태가 [Batch, SeqLen, Features] 임을 명시
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        """
        :param x: [B, L, N, F] 형태의 원본 시계열 피처
                  B(Batch), L(Lookback), N(Nodes), F(Features)
        :return: [B, N, hidden_dim] 형태의 시간 임베딩 (마지막 타임스텝의 GRU 은닉 상태)
        """
        B, L, N, F = x.shape
        
        # 노드를 Batch 차원처럼 병합: [B, L, N, F] -> [B, N, L, F] -> [B*N, L, F]
        # 즉 GRU는 '각 노드의 14일치 흐름'을 독립적인 시계열로 간주하고 인코딩합니다.
        # 주의: 축 변환(permute) 없이 view를 쓰면 시간과 노드 데이터가 섞이므로 반드시 permute 적용.
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N, L, F)
        
        # out: 모든 타임스텝의 은닉 상태, h_n: 마지막 은닉 상태 ([num_layers, B*N, hidden_dim])
        out, h_n = self.gru(x_reshaped)
        
        # 마지막 레이어(Layer)의 결과물만 사용: [B*N, hidden_dim]
        last_hidden = h_n[-1]
        
        # 다시 원래의 배치와 노드 차원으로 복구: [B, N, hidden_dim]
        node_embeddings = last_hidden.view(B, N, -1)
        
        return node_embeddings
