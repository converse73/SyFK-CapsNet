import torch
import torch.nn as nn
from typing import Tuple, Dict
from layers import PositionalEncoding, KANLayer, PrimaryCapsuleLayer, RoutingLayer

class TransformerEncoderSequence(nn.Module):
    def __init__(self, input_feature_dim: int, time_steps: int, output_hidden_size: int,
                 d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1,
                 lstm_num_layers: int = 1, lstm_bidirectional: bool = False):
        super().__init__()
        self.input_projection = nn.Linear(input_feature_dim, d_model) if input_feature_dim != d_model else nn.Identity()
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=time_steps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=lstm_num_layers, batch_first=True,
            bidirectional=lstm_bidirectional
        )
        lstm_output_dim = d_model * (2 if lstm_bidirectional else 1)
        self.output_projection = nn.Linear(lstm_output_dim, output_hidden_size)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x_seq)
        x = self.positional_encoding(x)

        encoded_sequence = self.transformer_encoder(x)

        _, (h_n, _) = self.lstm(encoded_sequence)

        if self.lstm.bidirectional:
            final_lstm_output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            final_lstm_output = h_n[-1, :, :]

        projected_features = self.output_projection(final_lstm_output)
        return projected_features

class SyKCABModel(nn.Module):
    def __init__(self, rs_input_feature_dim: int, m_input_feature_dim: int,
                 time_steps: int,
                 transformer_d_model: int, transformer_nhead: int,
                 transformer_num_encoder_layers: int, transformer_dim_feedforward: int,
                 transformer_dropout: float,
                 lstm_hidden_size: int, lstm_num_layers: int, lstm_bidirectional: bool,
                 num_rs_primary_caps: int, num_m_primary_caps: int, num_fused_primary_caps: int,
                 primary_cap_dim: int,
                 num_digit_caps: int, digit_cap_dim: int,
                 routing_iterations: int,
                 kan_params: Dict = None, use_kan_in_primary_caps: bool = True):
        super().__init__()
        kan_params = kan_params if kan_params is not None else {}

        self.rs_transformer_lstm_encoder = TransformerEncoderSequence(
            input_feature_dim=rs_input_feature_dim, time_steps=time_steps, output_hidden_size=lstm_hidden_size,
            d_model=transformer_d_model, nhead=transformer_nhead, num_encoder_layers=transformer_num_encoder_layers,
            dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout,
            lstm_num_layers=lstm_num_layers, lstm_bidirectional=lstm_bidirectional
        )
        self.m_transformer_lstm_encoder = TransformerEncoderSequence(
            input_feature_dim=m_input_feature_dim, time_steps=time_steps, output_hidden_size=lstm_hidden_size,
            d_model=transformer_d_model, nhead=transformer_nhead, num_encoder_layers=transformer_num_encoder_layers,
            dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout,
            lstm_num_layers=lstm_num_layers, lstm_bidirectional=lstm_bidirectional
        )

        transformer_output_dim = lstm_hidden_size

        self.kan_fusion_layer = KANLayer(
            width=[2 * transformer_output_dim, transformer_output_dim],
            **kan_params
        )

        self.rs_primary_caps = PrimaryCapsuleLayer(
            in_features=transformer_output_dim, num_caps=num_rs_primary_caps, cap_dim=primary_cap_dim,
            use_kan_linear=use_kan_in_primary_caps, kan_params=kan_params
        )
        self.m_primary_caps = PrimaryCapsuleLayer(
            in_features=transformer_output_dim, num_caps=num_m_primary_caps, cap_dim=primary_cap_dim,
            use_kan_linear=use_kan_in_primary_caps, kan_params=kan_params
        )
        self.fused_primary_caps = PrimaryCapsuleLayer(
            in_features=transformer_output_dim, num_caps=num_fused_primary_caps, cap_dim=primary_cap_dim,
            use_kan_linear=use_kan_in_primary_caps, kan_params=kan_params
        )

        total_primary_caps = num_rs_primary_caps + num_m_primary_caps + num_fused_primary_caps
        print(
            f"Capsule Network Design: Total {total_primary_caps} primary capsules ({num_rs_primary_caps} RS, {num_m_primary_caps} M, "
            f"{num_fused_primary_caps} Fused) will be routed to {num_digit_caps} advanced capsules.")

        self.routing_capsules = RoutingLayer(
            num_primary_caps=total_primary_caps, primary_cap_dim=primary_cap_dim,
            num_digit_caps=num_digit_caps, digit_cap_dim=digit_cap_dim,
            routing_iterations=routing_iterations
        )

        self.output_kan = KANLayer(
            width=[num_digit_caps, 1], **kan_params
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        rs_x, m_x = x

        rs_encoded_features = self.rs_transformer_lstm_encoder(rs_x)
        m_encoded_features = self.m_transformer_lstm_encoder(m_x)

        concatenated_features = torch.cat([rs_encoded_features, m_encoded_features], dim=1)
        fused_encoded_features = self.kan_fusion_layer(concatenated_features)

        rs_caps = self.rs_primary_caps(rs_encoded_features)
        m_caps = self.m_primary_caps(m_encoded_features)
        fused_caps = self.fused_primary_caps(fused_encoded_features)

        primary_caps_output = torch.cat([rs_caps, m_caps, fused_caps], dim=1)

        digit_caps_output = self.routing_capsules(primary_caps_output)

        digit_caps_lengths = torch.sqrt((digit_caps_output ** 2).sum(dim=-1))

        final_prediction = self.output_kan(digit_caps_lengths)

        return final_prediction