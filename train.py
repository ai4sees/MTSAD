from args import get_parser
from utils import *
from prediction import Predictor
from training import Trainer
import json
from datetime import datetime
import torch.nn as nn
import pickle
from tran_encs import TRAN_ENC



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
    print(args_summary)
    win_nht = {30: 5, 50: 10, 100: 20, 200: 25}
    n_head_t = win_nht[window_size]
    data_nhf = {"SMD": 19, "SMAP": 5, "MSL": 11}  # 38 - 2 heads, 25 / 5 -> 5 heads 55 / 11 -->
    n_head_f = data_nhf[dataset]
    n_feat_encs = int(args.arch.split('-')[0])
    n_time_encs = int(args.arch.split('-')[1])
    n_out_encs = int(args.arch.split('-')[2])


    id = datetime.now().strftime("%d%m%Y_%H%M%S")


    if dataset == 'SMD':
        output_path = f'output/SMD/{arch}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}/{arch}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    x_t = (x_train.shape[0]-window_size)//2560
    y_t = (x_test.shape[0]//256 - 1) * 256
    x_train = x_train[:x_t*2560+window_size]
    x_test = x_test[:y_t+window_size]
    y_test = y_test[:y_t+window_size]


    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

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
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        n_feat_encs = n_feat_encs,
        n_time_encs = n_time_encs,
        n_out_encs = n_out_encs,
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
    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=True)

    # Save dictionary to a Pickle file
    with open(f'{save_path}/losses.pkl', 'wb') as pickle_file:
        pickle.dump(trainer.losses, pickle_file)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )


    if dataset == "MSL":
        msl_flag = True
    else:
        msl_flag = False
    print(msl_flag)
    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label, msl_flag)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
