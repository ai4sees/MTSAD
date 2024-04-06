import json
from datetime import datetime
import torch
import torch.nn as nn
from args import get_parser
from architecture.utils import *
from evaluation.prediction import Predictor
from training import Trainer

from architecture.modules import (
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
        gru_n_layers=1,
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

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    arch = args.arch
    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    win_nht = {30: 5, 50: 10, 100: 20, 200: 25}
    n_head_t = win_nht[window_size]
    data_nhf = {"SMD": 19, "SMAP": 5, "MSL": 11}
    n_head_f = data_nhf[dataset]
    n_feat_encs = int(args.arch.split('-')[0])
    n_time_encs = int(args.arch.split('-')[1])
    n_out_encs = int(args.arch.split('-')[2])
    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Getting the dataset
    if dataset == 'SMD':
        output_path = f'output/SMD/{arch}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}/{arch}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')
    
    # Preprocessing
    x_t = (x_train.shape[0]-window_size)//2560
    y_t = (x_test.shape[0]//256 - 1) * 256
    x_train = x_train[:x_t*2560+window_size]
    x_test = x_test[:y_t+window_size]
    y_test = y_test[:y_t+window_size]

    # Making output paths
    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    # Splitting data into train and test
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    # Get number of Dimensions
    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    # Assign the train and test data with Sliding window
    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
        )
    
    model = TRAN_ENC(
        n_features,
        window_size,
        out_dim,
        n_head_f,
        n_head_t,
        kernel_size=args.kernel_size,
        # use_gatv2=args.use_gatv2,
        # feat_gat_embed_dim=args.feat_gat_embed_dim,
        # time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        # alpha=args.alpha
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )
    print(trainer.model)