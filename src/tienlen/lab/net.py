from itertools import pairwise

import torch
from torch import nn
from torch.nn.functional import relu

from tienlen.lab.interface import ModelSpec


def _layers(sizes: list[int]) -> nn.ModuleList:
    return nn.ModuleList(nn.Linear(a, b) for a, b in pairwise(sizes))


def _apply(layers: nn.ModuleList, values: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        values = relu(layer(values))
    return values


def segment_mean(values: torch.Tensor, segments: torch.Tensor, n_segments: int) -> torch.Tensor:
    totals = torch.zeros(n_segments, values.shape[1]).index_add(0, segments, values)
    counts = torch.zeros(n_segments).index_add(0, segments, torch.ones(len(segments)))
    return totals / counts[:, None]


class LabNet(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        state_sizes = [spec.state_dim, *spec.state_hidden]
        self.state = _layers(state_sizes)
        action_sizes = [spec.action_dim + state_sizes[-1], *spec.action_hidden]
        self.action = _layers(action_sizes)
        self.output = nn.Linear(action_sizes[-1] * (2 if spec.deep_set else 1), 1)
        self.value = nn.Linear(state_sizes[-1], 1) if spec.value else None

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, segments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = _apply(self.state, states)
        embedded = _apply(self.action, torch.cat([hidden[segments], actions], dim=1))
        if self.spec.deep_set:
            pooled = segment_mean(embedded, segments, len(states))
            embedded = torch.cat([embedded, pooled[segments]], dim=1)
        scores = self.output(embedded).squeeze(-1)
        return scores, None if self.value is None else self.value(hidden).squeeze(-1)


def export_weights(model: LabNet) -> dict:
    return {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
