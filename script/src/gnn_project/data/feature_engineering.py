import torch

from .dataset import OutletPredictionDataset


def _rolling_mean(features: torch.Tensor, window: int) -> torch.Tensor:
    """Causal rolling mean over the time axis."""
    total_steps = features.shape[0]
    out = torch.zeros_like(features)
    for t in range(total_steps):
        start = max(0, t - window + 1)
        out[t] = features[start : t + 1].mean(dim=0)
    return out


def augment_node_features(
    node_features: torch.Tensor,
    *,
    include_delta: bool = False,
    rolling_windows=(),
    keep_node_type: bool = True,
) -> torch.Tensor:
    """
    Expand [T, N, F] node features with causal engineered channels.

    Expected convention:
    - last channel is the node-type flag
    - engineered features are computed on every non-node-type channel only
    """
    if not torch.is_tensor(node_features):
        node_features = torch.as_tensor(node_features, dtype=torch.float32)

    if keep_node_type:
        base = node_features[..., :-1]
        node_type = node_features[..., -1:]
    else:
        base = node_features
        node_type = None

    blocks = [base]

    if include_delta:
        delta = torch.zeros_like(base)
        delta[1:] = base[1:] - base[:-1]
        blocks.append(delta)

    for window in rolling_windows:
        blocks.append(_rolling_mean(base, window=window))

    if node_type is not None:
        blocks.append(node_type)

    return torch.cat(blocks, dim=-1)


def clone_dataset_with_features(dataset: OutletPredictionDataset, new_node_features: torch.Tensor, name: str):
    """Rebuild a dataset with engineered features but the same targets/lookback."""
    return OutletPredictionDataset(
        node_features=new_node_features.detach().cpu().numpy(),
        outlet_target=dataset.targets.detach().cpu().numpy(),
        lookback_window=dataset.lookback_window,
        name=name,
    )


def build_feature_variant_datasets(
    train_ds: OutletPredictionDataset,
    val_ds: OutletPredictionDataset,
    test_ds: OutletPredictionDataset,
    *,
    include_delta: bool = False,
    rolling_windows=(),
    variant_name: str = "baseline",
):
    train_features = augment_node_features(
        train_ds.node_features,
        include_delta=include_delta,
        rolling_windows=rolling_windows,
    )
    val_features = augment_node_features(
        val_ds.node_features,
        include_delta=include_delta,
        rolling_windows=rolling_windows,
    )
    test_features = augment_node_features(
        test_ds.node_features,
        include_delta=include_delta,
        rolling_windows=rolling_windows,
    )

    return (
        clone_dataset_with_features(train_ds, train_features, name=f"Train-{variant_name}"),
        clone_dataset_with_features(val_ds, val_features, name=f"Val-{variant_name}"),
        clone_dataset_with_features(test_ds, test_features, name=f"Test-{variant_name}"),
        train_features.shape[-1],
    )
