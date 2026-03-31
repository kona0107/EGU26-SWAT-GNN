import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SpatialEncoder(nn.Module):
    """
    공간 인코더 (GAT 모듈).
    상하류 연결망(Edge Index)을 따라 정보를 전파하고 어텐션 가중치(Attention Weight)를 학습합니다.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        # concat=True 이므로 다음 레이어의 입력 크기는 (hidden_channels * heads)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # 마지막 GAT 병합을 위함
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """
        :param x: [B*N, in_channels] 형태의 시간 인코딩(임베딩)된 노드 리스트
        :param edge_index: [2, E*B] 형태의 '배치(Batch)'로 확장된 연결 구조 행렬
        :return: [B*N, out_channels] 형태의 공간적 특징 묶임
        """
        # 첫 번째 GAT 통과
        out1 = self.gat1(x, edge_index)
        out1 = self.relu(out1)
        
        # 두 번째 GAT 통과로 정보를 정제 (GATConv 겹칩)
        out2 = self.gat2(out1, edge_index)
        
        return out2
