import json
from datetime import datetime
import torch
import torch.nn as nn
from utils import *
from models import GRUAT
from arguments import get_parser
from trainer import Trainer
from anomaly_detection import AnomalyDetector


from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

fold_losses_log = {}

def train_and_evaluate_model(X_fold, y_fold, hyperparams):
        
        val_split = 0.2
        # Unpack hyperparameters
        batch_size = hyperparams['batch_size']
        window_size = hyperparams['window_size']
        n_epochs = hyperparams['n_epochs']
        # init_lr = hyperparams['init_lr']
        # node_embed_dim = window_size

        X_fold_torch = torch.tensor(X_fold, dtype=torch.float)


        # Data loading and preparation as per your original code
      
        train_dataset = SlidingWindowDataset2(X_fold_torch, window_size, target_dims) if dataset.upper() == 'CUSTOM' else SlidingWindowDataset(x_train, window_size, target_dims)
        # test_dataset = SlidingWindowDataset2(x_test, window_size, target_dims) if dataset.upper() == 'CUSTOM' else SlidingWindowDataset(x_test, window_size, target_dims)

        train_loader, val_loader, test_loader = generate_data_loaders(
            train_dataset, batch_size, val_split, shuffle_dataset
        )

    
        # model = GRUAT(n_features, out_dim, window_size, hyperparams['memory_dim'], hyperparams['num_memory_slots'], args.node_embed_dim, args.num_heads,
        #                 hyperparams['gru_num_layers'], hyperparams['gru_hid_dim'], args.tcn_embed_dim, args.tcn_kernel_size, hyperparams['fc_n_layers'], 
        #                 hyperparams['gru_hid_dim'], args.vae_latent_dim, args.recon_hid_dim, args.recon_n_layers, hyperparams['dropout'], args.alpha)
        
        model = GRUAT(n_features, out_dim, window_size, args.memory_dim, args.num_memory_slots, window_size, args.num_heads,
                        args.gru_num_layers, args.gru_hidden_dim, args.tcn_embed_dim, args.tcn_kernel_size, args.fc_n_layers, 
                        hyperparams['fc_hid_dim'], args.vae_latent_dim, args.recon_hid_dim, args.recon_n_layers, hyperparams['dropout'], args.alpha)
 
        # Define loss and optimizer
        criterion = nn.MSELoss()  # For regression
        # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-5)



        # Initialize trainer
        trainer = Trainer(
            model=model,
            target_dim = target_dims,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            save_path=model_save_path
        )

        trainer.train(n_epochs)

        # Evaluate the model
        test_loss = trainer.validate(val_loader)
        
        # Return test loss (or any metric you want to optimize)
        return test_loss[2]  # Return total test loss

# param_distributions = {
#     'batch_size': [32, 64, 128, 256],        # Typically 256+ can be large for GRU-based models
#     'window_size': [50, 100, 150],           # Reasonable for short vs. medium sequences
#     'n_epochs': [5, 30, 50, 100],          # Fewer than 30 often insufficient, >100 can be tested if time allows
#     'init_lr': [1e-4, 1e-5],      # 1e-2 can be too big for many RNN/attention combos
#     'dropout': [0.1, 0.2, 0.4, 0.5],         # 0.3 was fine, but 0.4 often helps deeper or larger models
#     'memory_dim': [16, 32, 64],         # 8 might be too small to store rich patterns
#     'num_memory_slots': [8, 16, 32],    # 64 can be large; you can bring it back if you have big data
#     'gru_num_layers': [1, 2, 3],        # 5 might be excessive unless you have large data/time
#     'gru_hid_dim': [64, 128, 150],      # 150 is your baseline, also try smaller/larger for coverage
#     'fc_n_layers': [1, 2, 3]           # 5 can be quite deep; 1-3 often suffice
#     # 'fc_hid_dim': [64, 128, 150]       # Similarly, vary this to see if smaller/larger FC helps
#     }
# #     'recon_n_layers': [1, 2, 3],        # Keep it simpler than 5, unless you specifically need deeper
# #     'recon_hid_dim': [64, 128, 150]     # Mirror the logic for fc_hid_dim
# # }

param_distributions = {
      # Typically 256+ can be large for GRU-based models
    'n_epochs': [10, 30, 50],
    'batch_size': [64, 128, 256],
    'window_size': [50, 100, 150],           # Reasonable for short vs. medium sequences
    # 'memory_dim': [16, 32],         # 8 might be too small to store rich patterns
    # 'num_memory_slots': [8, 16, 32],    # 64 can be large; you can bring it back if you have big data
    'fc_hid_dim': [64, 128, 150],
    'dropout': [0.3, 0.4]
    }

class GRUATModel(BaseEstimator):
    def __init__(self, batch_size, window_size, n_epochs, init_lr, dropout, memory_dim, num_memory_slots, gru_num_layers, gru_hid_dim, fc_n_layers, fc_hid_dim, recon_n_layers, recon_hid_dim):
        self.batch_size = batch_size
        self.window_size = window_size
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.dropout = dropout
        self.memory_dim = memory_dim
        self.num_memory_slots = num_memory_slots
        self.gru_num_layers  =  gru_num_layers
        self.gru_hid_dim = gru_hid_dim
        self.fc_n_layers  = fc_n_layers
        self.fc_hid_dim  = fc_hid_dim
        self.recon_n_layers  = recon_n_layers
        self.recon_hid_dim  = recon_hid_dim



    def fit(self, X, y=None):
        hyperparams = {
            'batch_size': self.batch_size,
            'window_size': self.window_size,
            'n_epochs': self.n_epochs,
            'init_lr': self.init_lr,
            'dropout': self.dropout,         
            'memory_dim': self.memory_dim,          
            'num_memory_slots': self.num_memory_slots,    
            'gru_num_layers': self.gru_num_layers,          
            'gru_hid_dim': self.gru_hid_dim,      
            'fc_n_layers': self.fc_n_layers,    
            'fc_hid_dim': self.fc_hid_dim
        }   
            # 'recon_n_layers': self.recon_n_layers,      
            # 'recon_hid_dim': self.recon_hid_dim
            #     }
        print("Current Hyperparameter: ", hyperparams)
        X_train_fold, X_val_fold = train_test_split(X, test_size=0.2, shuffle=False)  # Do a separate split strategy for custom dataset
        final_loss = train_and_evaluate_model(X_train_fold, X_val_fold, hyperparams)


        # final_loss = train_and_evaluate_model(hyperparams)
         # 2) If NaN, penalize with a large numeric value or skip

        if torch.isnan(torch.tensor(final_loss)):
            final_loss = 1e12  # penalize heavily
                # raise ValueError("NaN encountered for this hyperparam combination")

            # 3) Store this loss on the estimator
        self._final_loss = final_loss

         # ---------------------------------------------------------------------
        #  STORE THE LOSS IN OUR GLOBAL fold_losses_log
        # ---------------------------------------------------------------------
        param_key = str(sorted(hyperparams.items()))
        if param_key not in fold_losses_log:
            fold_losses_log[param_key] = []
        fold_losses_log[param_key].append(final_loss)

            # 4) Return the estimator itself
        return self

    def score(self, X, y=None):
        # scikit-learn calls this after fit, expecting a higher-is-better score
        # We want to minimize loss, so return negative
        return -self._final_loss

if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Number of GPUs available:", torch.cuda.device_count())

    if dataset == 'CUSTOM':
        output_path = f'output/Custom'
        data_save_path = path.join("datasets", dataset, 'processed', "processed.pkl") 
    elif dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        data_save_path = path.join("datasets", dataset, "processed", f"machine-{group_index}-{index}_processed.pkl")
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/data'
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
    
    if dataset == 'CUSTOM':
        x_train = x_train.float()
        x_test = x_test.float()
        # y_test = y_test.float()

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

        # Now define the search object
    mtad_gat_model = GRUATModel(batch_size=None, window_size=None, n_epochs=None, init_lr=None,
    dropout=None, memory_dim = None, num_memory_slots=None, gru_num_layers=None, gru_hid_dim= None, 
    fc_n_layers=None, fc_hid_dim= None, recon_n_layers=None, recon_hid_dim=None)

    random_search = RandomizedSearchCV(
        mtad_gat_model, 
        param_distributions=param_distributions, 
        n_iter=50,  # Number of random configurations to try
        scoring='neg_mean_squared_error',  # Optimize for lower loss
        cv=2,  # 3-fold cross-validation
        verbose=2,
        random_state=42
    )


    # Start random search
    random_search.fit(x_train)
    cv_results = random_search.cv_results_

    json_save_path = 'HP_Logs'
    save_dir = f'{json_save_path}/{id}'
    if not path.exists(save_dir):
        makedirs(save_dir)
    file_path = path.join(save_dir, f"hp_search_{id}.json")

    with open(file_path, 'w') as f:
        json.dump({
            'params': [str(p) for p in cv_results['params']],
            'mean_test_score': cv_results['mean_test_score'].tolist(),
            'std_test_score': cv_results['std_test_score'].tolist(),
            'rank_test_score': cv_results['rank_test_score'].tolist(),
        }, f, indent=4)

    print("Best hyperparameters found:", random_search.best_params_)
    print("Best score (negative loss):", random_search.best_score_)

    fold_loss_summary = []
    for param_key, losses_list in fold_losses_log.items():
            avg_loss = sum(losses_list) / len(losses_list)
            fold_loss_summary.append({
                'hyperparams': param_key,
                'fold_losses': losses_list,
                'mean_final_loss': avg_loss
            })
    
    custom_file_path = path.join(save_dir, f"custom_hp_log_{id}.json")

    with open(custom_file_path, 'w') as f:
            json.dump(fold_loss_summary, f, indent=4)

    print("\n===== Custom fold_losses_log =====")
    for entry in fold_loss_summary:
            print(entry)