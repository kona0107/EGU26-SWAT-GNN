import torch
import torch.nn as nn
from .temporal import TemporalEncoder, GRUTemporalEncoder
from .gcn import SpatialEncoder

class SpatioTemporalHybridGNN(nn.Module):
    """
    Model 2. Temporal + SAGEConv (시공간 하이브리드 모델)
    - 시간적 정보: 모든 노드가 각각 인코더(Transformer/GRU)를 거쳐 임베딩을 생성
    - 공간적 정보: SAGEConv를 통해 상류의 임베딩이 하류로 전파됨
    - 예측 정보: GNN 통과 후, 유출구 노드의 Hidden State만 읽어서 예측
    """
    def __init__(self, in_features=11, temporal_hidden=32, gcn_hidden=16, 
                 out_features=1, num_temporal_layers=2, temporal_type="transformer"):
        super().__init__()
        
        self.temporal_type = temporal_type.lower()
        if self.temporal_type == "transformer":
            self.temporal_encoder = TemporalEncoder(
                input_dim=in_features, 
                hidden_dim=temporal_hidden, 
                num_layers=num_temporal_layers
            )
        elif self.temporal_type == "gru":
            self.temporal_encoder = GRUTemporalEncoder(
                input_dim=in_features, 
                hidden_dim=temporal_hidden, 
                num_layers=num_temporal_layers
            )
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")
        
        self.spatial_encoder = SpatialEncoder(
            in_channels=temporal_hidden, 
            hidden_channels=gcn_hidden, 
            out_channels=gcn_hidden
        )
        
        # MLP Readout Layer
        self.predictor = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden // 2),
            nn.ReLU(),
            nn.Linear(gcn_hidden // 2, out_features)
        )

    def forward(self, x_seq, edge_index, outlet_node_idx=-1):
        """
        :param x_seq: [B, L, N, F] (과거 윈도우로 구성된 입력)
        :param edge_index: [2, E] (반드시 Source->Target 방향의 Directed Edge여야 하며, .to_undirected() 호출 금지)
        :param outlet_node_idx: 유출구 노드 번호 (보통 맨 마지막 노드)
        """
        B, L, N, F = x_seq.shape
        
        # 1. Temporal Encode (각 노드별로 독립적인 시간적 패턴 학습) => [B, N, temporal_hidden]
        node_embeds = self.temporal_encoder(x_seq)
        
        # 2. PyG의 GNN 층에 넣기 위해 Batch와 Node 차원을 1자로 폅니다 => [B*N, temporal_hidden]
        node_embeds_flat = node_embeds.view(B * N, -1)
        
        # 3. 모델 안의 각 배치 데이터가 서로의 그래프를 침범하지 못하도록 Edge Index 복제 (Batching 기법)
        offset = torch.arange(0, B * N, N, device=edge_index.device).view(-1, 1, 1)
        batched_edge_index = edge_index.unsqueeze(0) + offset  # [B, 2, E]
        batched_edge_index = batched_edge_index.transpose(0, 1).reshape(2, -1) # 최종형태: [2, B*E]
        
        # 4. 상류 -> 하류 공간적 정보 전파 => [B*N, gcn_hidden]
        spatial_out_flat = self.spatial_encoder(node_embeds_flat, batched_edge_index)
        
        # 5. 펼쳤던 차원을 본래의 배치, 노드 단위로 복구 => [B, N, gcn_hidden]
        spatial_out = spatial_out_flat.view(B, N, -1)
        
        # 6. 유출구(Outlet Node) Readout
        # 상류 정보가 융합된 유출구의 최종 임베딩만 단독 추출 => [B, gcn_hidden]
        outlet_embed = spatial_out[:, outlet_node_idx, :]
        
        # 7. 단일 타겟 지표 예측 (Chl-a 도출) => [B, out_features]
        preds = self.predictor(outlet_embed)
        
        return preds
