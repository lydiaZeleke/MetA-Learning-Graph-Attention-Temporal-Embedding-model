from datetime import datetime
import json
import numpy as np
from os import listdir, makedirs, path

import torch
import torch.nn as nn
from utils import *
from models import GRUAT
from arguments import get_parser
from trainer import Trainer
from anomaly_detection import AnomalyDetector

if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    meta_mode = args.meta_training
    dataset = args.dataset
    window_size = args.lookback
    group_index = args.group[0]
    index = args.group[2:]
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    patience = args.patience
    min_delta = args.min_delta
    train_adaptation_steps = args.train_adaptation_steps
    test_adaptation_steps = args.test_adaptation_steps
    inner_lr = args.inner_lr
    meta_lr = args.meta_lr
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == 'CUSTOM':
        output_path = f'output/Custom'
        data_save_path = path.join("datasets", dataset, 'processed', "processed.pkl") 
    elif dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        data_save_path = path.join("datasets", dataset, "processed", f"machine-{group_index}-{index}_processed.pkl")
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/data/'
        data_save_path = path.join("datasets", "data", "processed", f"{dataset}_processed.pkl") 
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    
    log_dir = f'{output_path}/logs'
    if not path.exists(output_path):
        makedirs(output_path)
    if not path.exists(log_dir):
        makedirs(log_dir)
    model_save_path = f"{output_path}/{id}"

    x_train, x_test, y_test = load_tensors_from_pickle(data_save_path)
    
    if torch.isnan(x_train).any():
            print('NaN values detected in input')
    if torch.isnan(x_test).any():
            print('NaN values detected in input')
    if dataset == 'CUSTOM':
        x_train = x_train.float()
        x_test = x_test.float()
 
    n_features = x_train.shape[1]
    output_size = n_features

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

     #''''''''''' META-LEARNING IMPLEMENTATION '''''''''''''#
    if meta_mode:
        train_loaders, test_loaders, label_loaders = prepare_dataloaders(x_train, x_test, y_test, window_size, target_dims, batch_size)
        val_loaders = None

     #''''''''''' BASE MODEL IMPLEMENTATION '''''''''''''#
    else:
        train_dataset = SlidingWindowDataset2(x_train, window_size, target_dims) if dataset.upper() == 'CUSTOM' else SlidingWindowDataset(x_train, window_size, target_dims)
        test_dataset = SlidingWindowDataset2(x_test, window_size, target_dims) if dataset.upper() == 'CUSTOM' else SlidingWindowDataset(x_test, window_size, target_dims)

        train_loaders, val_loaders, test_loaders = generate_data_loaders(
            train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
        )
 
    # model = GRUAT(n_features, args.gru_hidden_dim, args.gru_num_layers, output_size, window_size, args.dropout, args.alpha, args.gru_final_hid_dim)
    
    model = GRUAT(n_features, out_dim, window_size, args.memory_dim, args.num_memory_slots, args.node_embed_dim, 
                    args.gru_num_layers, args.gru_hidden_dim, args.tcn_embed_dim, args.tcn_kernel_size, args.fc_n_layers, 
                    args.fc_hid_dim, args.vae_latent_dim, args.recon_hid_dim, args.recon_n_layers, args.dropout, args.alpha)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()  # For regression
    # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-5)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        target_dim = target_dims,
        train_loader=train_loaders,
        val_loader=val_loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_path=model_save_path,
        inner_lr=inner_lr,
        meta_lr= meta_lr
    )


   
    if meta_mode:  #''''''''''' META-LEARNING IMPLEMENTATION - TASK-SPECIFIC LEARNING AND META-TRAINING'''''''''''''#
        trainer.meta_train(train_loaders, n_epochs, train_adaptation_steps)
    else:
         trainer.train(n_epochs)

    #''''''''''' ''''''''''' '''''''''''''#


    ########################

    
    # plot_losses(trainer.losses, save_path=save_path, plot=False)
    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "CUSTOM": (0.9925, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
        
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "CUSTOM":1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]
  
   
    if not meta_mode:
    # Check test loss
     #''''''''''' BASE MODEL IMPLEMENTATION '''''''''''''#
        test_loss = trainer.validate(test_loaders)

        print(f"Test forecast loss: {test_loss[0]:.5f}")
        print(f"Test reconstruction loss: {test_loss[1]:.5f}")
        print(f"Test total loss: {test_loss[2]:.5f}")

        trainer.load(f"{model_save_path}/model.pt")
        best_model = trainer.model
    #''''''''''' ''''''''''' '''''''''''''#

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
        "save_path": model_save_path
    }

    
    if meta_mode: #''''''''''' META-TESTING IMPLEMENTATION'''''''''''''#
        anomaly_detector_args = [window_size, n_features, prediction_args]
        evaluation_results = trainer.meta_test(anomaly_detector_args, test_loaders, label_loaders, test_adaptation_steps)

        print(evaluation_results)

    else:     #''''''''''' BASE MODEL IMPLEMENTATION '''''''''''''#
        predictor = AnomalyDetector(
            best_model,
            window_size,
            n_features,
            prediction_args,
        )

        label = y_test[window_size:] if y_test is not None else None
        predictor.predict_anomalies(x_train, x_test, label)
    #''''''''''' ''''''''''' '''''''''''''#
