import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class SpatialEncoder(nn.Module):
    """
    SAGEConv 기반의 공간 정보 병합 모듈.
    상류(Source)에서 하류(Target)로 정보가 흐르는 특이성을 안전하게 반영합니다.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv는 Target 노드 자신의 Self-information과 
        # Source (상류) 노드들의 정보들을 집계하여 업데이트합니다.
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x
