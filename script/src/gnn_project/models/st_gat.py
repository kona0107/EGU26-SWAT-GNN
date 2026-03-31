import torch
import torch.nn as nn
from .temporal import TemporalEncoder
from .gat import SpatialEncoder

class SpatioTemporalGNN(nn.Module):
    """
    Temporal-first, Spatial-second 구조의 통합 시공간 그래프 메인 모델.
    주 타깃인 위성 Chl-a 예측에 최적화.
    """
    def __init__(self, in_features, gru_hidden, gat_hidden, out_features=1, num_gru_layers=1, gat_heads=2):
        super().__init__()
        
        # 1. Temporal Component (시간 우선 처리)
        self.temporal_encoder = TemporalEncoder(
            input_dim=in_features, 
            hidden_dim=gru_hidden, 
            num_layers=num_gru_layers
        )
        
        # 2. Spatial Component (공간 관계 전파)
        self.spatial_encoder = SpatialEncoder(
            in_channels=gru_hidden, 
            hidden_channels=gat_hidden, 
            out_channels=gat_hidden, 
            heads=gat_heads
        )
        
        # 3. Predictor 블록 (마지막 타겟 수치 매핑)
        self.predictor = nn.Sequential(
            nn.Linear(gat_hidden, gat_hidden // 2),
            nn.ReLU(),
            nn.Linear(gat_hidden // 2, out_features)
        )

    def forward(self, x_seq, base_edge_index):
        """
        :param x_seq: [B, L, N, F] (과거 윈도우 + 노드 + 변수배열)
        :param base_edge_index: [2, E] Subbasin 간 상류->하류 물리적 연결 (고정값)
        :return: [B, N, out_features] 형태의 예측 행렬 (추후 마스크와 Element-wise 비교를 위함)
        """
        B, L, N, F = x_seq.shape
        
        # Step 1: 시간 우선 인코딩 => [B, N, gru_hidden]
        node_embeds = self.temporal_encoder(x_seq)
        
        # Step 2: GAT는 [B*N, 피쳐] 형태를 입력받으므로 차원 Flat 처리
        node_embeds_flat = node_embeds.view(B * N, -1)
        
        # Step 3: Base Edge Index를 현재 배치 사이즈 단위인 B 크기에 맞춰 복제 연산
        # (PyG에서는 이를 Block Diagonal 형태로 구성해 상호 배치(batch) 간 간섭을 차단합니다)
        offset = torch.arange(0, B * N, N, device=base_edge_index.device).view(-1, 1, 1)
        batched_edge_index = base_edge_index.unsqueeze(0) + offset  # [B, 2, E]
        batched_edge_index = batched_edge_index.transpose(0, 1).reshape(2, -1) # 최종: [2, B*E]
        
        # Step 4: 공간적 인코딩 (상하류 어텐션 학습) => [B*N, gat_hidden]
        spatial_out = self.spatial_encoder(node_embeds_flat, batched_edge_index)
        
        # Step 5: 최종 레이어 통과 (예측값 산출) => [B*N, out_features]
        preds_flat = self.predictor(spatial_out)

        # Step 6: 다시 원래 배치 및 노드 차원으로 복구 => [B, N, out_features]
        preds = preds_flat.view(B, N, -1)
        
        return preds
