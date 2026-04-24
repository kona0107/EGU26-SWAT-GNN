import torch
import torch.nn as nn

from .gcn import SpatialEncoder


def _build_batched_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    """Repeat a single graph edge list for a mini-batch of disjoint graphs."""
    offset = torch.arange(0, batch_size * num_nodes, num_nodes, device=edge_index.device).view(-1, 1, 1)
    batched_edge_index = edge_index.unsqueeze(0) + offset
    return batched_edge_index.transpose(0, 1).reshape(2, -1)


class FTTransformerTemporalEncoder(nn.Module):
    """
    Encode each node's feature-history table with an FT-Transformer style block.

    For each node we treat each feature as a token, and the token content is the
    lookback trajectory of that feature. A transformer then mixes feature tokens
    so the model can learn cross-feature interactions from the temporal window.
    """

    def __init__(
        self,
        input_dim: int,
        lookback_window: int,
        hidden_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lookback_window = lookback_window

        self.feature_projection = nn.Linear(lookback_window, hidden_dim)
        self.feature_token_embedding = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: [B, L, N, F]
        Returns:
            node_embeddings: [B, N, hidden_dim]
        """
        batch_size, lookback, num_nodes, num_features = x_seq.shape
        if lookback != self.lookback_window:
            raise ValueError(
                f"FTTransformerTemporalEncoder expected lookback={self.lookback_window}, "
                f"but got sequence length={lookback}."
            )

        # [B, L, N, F] -> [B, N, F, L] -> [B*N, F, L]
        feature_histories = x_seq.permute(0, 2, 3, 1).contiguous().view(batch_size * num_nodes, num_features, lookback)
        tokens = self.feature_projection(feature_histories)
        tokens = tokens + self.feature_token_embedding[:, :num_features, :]
        tokens = self.input_norm(tokens)

        cls_token = self.cls_token.expand(batch_size * num_nodes, -1, -1)
        transformer_input = torch.cat([cls_token, tokens], dim=1)
        transformer_output = self.transformer(transformer_input)

        node_embeddings = self.output_norm(transformer_output[:, 0, :])
        return node_embeddings.view(batch_size, num_nodes, -1)


class _BaseGraphPredictor(nn.Module):
    def __init__(self, temporal_hidden: int, gcn_hidden: int, out_features: int = 1):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(
            in_channels=temporal_hidden,
            hidden_channels=gcn_hidden,
            out_channels=gcn_hidden,
        )
        self.predictor = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gcn_hidden, out_features),
        )

    def _encode_graph(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_embeddings.shape
        flat_embeddings = node_embeddings.view(batch_size * num_nodes, -1)
        batched_edge_index = _build_batched_edge_index(edge_index, batch_size, num_nodes)
        graph_embeddings = self.spatial_encoder(flat_embeddings, batched_edge_index)
        return graph_embeddings.view(batch_size, num_nodes, -1)


class GCNFTTransformerModel(_BaseGraphPredictor):
    """GCN + FT-Transformer baseline."""

    def __init__(
        self,
        in_features: int,
        lookback_window: int,
        temporal_hidden: int = 64,
        gcn_hidden: int = 32,
        out_features: int = 1,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(temporal_hidden=temporal_hidden, gcn_hidden=gcn_hidden, out_features=out_features)
        self.temporal_encoder = FTTransformerTemporalEncoder(
            input_dim=in_features,
            lookback_window=lookback_window,
            hidden_dim=temporal_hidden,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor, outlet_node_idx: int = -1) -> torch.Tensor:
        node_embeddings = self.temporal_encoder(x_seq)
        graph_embeddings = self._encode_graph(node_embeddings, edge_index)
        outlet_embedding = graph_embeddings[:, outlet_node_idx, :]
        return self.predictor(outlet_embedding)


class PersistenceResidualGCNFTTransformerModel(GCNFTTransformerModel):
    """
    Predict a residual on top of the latest outlet chl value.

    The notebook feeds standardized targets, so the last outlet chl feature is on
    the same scale as the target as long as the outlet feature/target scalers are
    fitted on the same raw chl column.
    """

    def __init__(self, chl_feature_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.chl_feature_idx = chl_feature_idx

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor, outlet_node_idx: int = -1) -> torch.Tensor:
        residual = super().forward(x_seq, edge_index, outlet_node_idx=outlet_node_idx)
        last_outlet_chl = x_seq[:, -1, outlet_node_idx, self.chl_feature_idx].unsqueeze(-1)
        return last_outlet_chl + residual
