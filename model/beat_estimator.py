import torch
import torch.nn as nn
import torch.nn.functional as F


class BeatDistanceEstimator(nn.Module):
    def __init__(
        self,
        input_dim=24 * 3,
        hidden_dim=128,
        num_heads=4,
        num_layers=6,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joints):
        if joints.ndim != 4 or joints.shape[-2:] != (24, 3):
            raise ValueError("Expected joints with shape [B, T, 24, 3]")

        velocity = torch.zeros_like(joints)
        velocity[:, 1:] = joints[:, 1:] - joints[:, :-1]
        tokens = velocity.reshape(joints.shape[0], joints.shape[1], -1)
        hidden = self.input_proj(tokens)
        encoded = self.encoder(hidden)
        distances = self.output_head(encoded).squeeze(-1)
        return F.softplus(distances)


class G1BeatDistanceEstimator(nn.Module):
    def __init__(
        self,
        input_dim=38,
        hidden_dim=128,
        num_heads=4,
        num_layers=6,
        ff_dim=512,
        dropout=0.1,
        output_activation="softplus",
    ):
        super().__init__()
        if input_dim != 38:
            raise ValueError("G1BeatDistanceEstimator expects input_dim=38")
        if output_activation not in ("softplus", "sigmoid"):
            raise ValueError("output_activation must be 'softplus' or 'sigmoid'")
        self.input_dim = input_dim
        self.output_activation = output_activation
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pose):
        if pose.ndim != 3 or pose.shape[-1] != self.input_dim:
            raise ValueError("Expected normalized G1 pose with shape [B, T, 38]")

        velocity = torch.zeros_like(pose)
        velocity[:, 1:] = pose[:, 1:] - pose[:, :-1]
        hidden = self.input_proj(velocity)
        encoded = self.encoder(hidden)
        distances = self.output_head(encoded).squeeze(-1)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(distances)
        return F.softplus(distances)
