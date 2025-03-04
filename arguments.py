import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data params ---
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument(
        "--normalize_technique",
        choices=['min_max', 'z-score'],  # Restrict to these two values
        required=False,
        default = 'min_max',  
        help="Normalization technique to use: 'min_max' or 'z-score'."
    )
    parser.add_argument("--dataset", type=str.upper, default="SMD")
    parser.add_argument("--group", type=str, default="1-1", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=50) #100
    parser.add_argument("--spec_res", type=str2bool, default=False)

    # -- Model params ---

    # TCN parameters
    parser.add_argument("--tcn_embed_dim", type=int, default = 32)
    parser.add_argument("--tcn_kernel_size", type=int, default = 7)

    # GRU layers
    parser.add_argument("--gru_num_layers", type=int, default=3) #3
    parser.add_argument("--gru_hidden_dim", type=int, default=100) 

    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=1)
    # parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    
       ## ------External Memory and Fusion params ---------
    parser.add_argument("--memory_dim", type=int, default = 32) #32
    parser.add_argument("--num_memory_slots", type=int, default= 8) #16
    parser.add_argument("--node_embed_dim", type=int, default = 50) # node embedding made to match the window size for our current architectural setup
    
    # # Forecasting Model
    parser.add_argument("--fc_n_layers", type=int, default=3)#3
    parser.add_argument("--fc_hid_dim", type=int, default=128) #150 - must be same as gru_hid_dim
    # # Reconstruction Model
    parser.add_argument("--vae_latent_dim", type=int, default=32)
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    parser.add_argument("--recon_n_layers", type=int, default=3)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)

    # # --- Train params ---
    parser.add_argument("--epochs", type=int, default= 10) #10
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=64) #256
    parser.add_argument("--init_lr", type=float, default=1e-5) #1e-4
    parser.add_argument("--shuffle_dataset", type=str2bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.3) #0.3
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--patience", type=int, default=5) #256
    parser.add_argument("--min_delta", type=int, default=0.001) #256


    # # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    return parser