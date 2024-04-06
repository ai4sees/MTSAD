import torch
import torch.nn as nn



from modules import (
    ConvLayer,Conv1dLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
    Transformer_encoder
)
class TRAN_ENC(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        n_head_f,
        n_head_t,
        kernel_size=7,
        gru_n_layers= 1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        n_feat_encs = 1,
        n_time_encs = 1,
        n_out_encs = 1,
    ):
        super(TRAN_ENC, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.n_feat_encs = n_feat_encs
        self.n_time_encs = n_time_encs
        self.n_out_encs = n_out_encs

        # Feature-level transformer encoders
        self.feature_encoders = nn.ModuleList([
            Transformer_encoder(n_features, n_head_f, sequence_length=window_size, output_dim=n_features)
            for _ in range(self.n_feat_encs)
        ])

        # Time-level transformer encoders
        self.time_encoders = nn.ModuleList([
            Transformer_encoder(window_size, n_head_t, sequence_length=n_features, output_dim=window_size)
            for _ in range(self.n_time_encs)
        ])

        # Output-level transformer encoders
        self.output_encoders = nn.ModuleList([
            Transformer_encoder(3 * n_features, 3 * n_head_f, sequence_length=window_size, output_dim=3 * n_features)
            for _ in range(self.n_out_encs)
        ])

        self.conv1d = Conv1dLayer(3 * n_features, 1)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(150, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, 150, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        x = self.conv(x)
        inp1 = x.permute(1, 0, 2)  # F x T
        inp2 = x.permute(2, 0, 1)  # T x F

        # Process feature-level inputs through the feature encoders
        for encoder in self.feature_encoders:
            inp1 = encoder(inp1)

        # Process time-level inputs through the time encoders
        for encoder in self.time_encoders:
            inp2 = encoder(inp2)

        h_feat = inp1.permute(1, 0, 2)
        h_temp = inp2.permute(1, 2, 0)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)
        h_cat = self.conv1d(h_cat)

        # Process concatenated features through the output encoders
        h_cat = h_cat.permute(1, 0, 2)
        for encoder in self.output_encoders:
            h_cat = encoder(h_cat)

        h_cat = h_cat.permute(1, 0, 2)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons
