import json
from datetime import datetime
import torch
import torch.nn as nn
from utils import *
from model import GRUAT
from arguments import get_parser
from trainer import Trainer
from anomaly_detection import AnomalyDetector


from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


def train_and_evaluate_model(hyperparams):
        # Unpack hyperparameters
        batch_size = hyperparams['batch_size']
        window_size = hyperparams['window_size']
        n_epochs = hyperparams['n_epochs']
        init_lr = hyperparams['init_lr']
        # node_embed_dim = window_size

        # Data loading and preparation as per your original code
      
        train_dataset = SlidingWindowDataset2(x_train, window_size, target_dims) if dataset.upper() == 'CUSTOM' else SlidingWindowDataset(x_train, window_size, target_dims)
        test_dataset = SlidingWindowDataset2(x_test, window_size, target_dims) if dataset.upper() == 'CUSTOM' else SlidingWindowDataset(x_test, window_size, target_dims)

        train_loader, val_loader, test_loader = generate_data_loaders(
            train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
        )
        
        model = GRUAT(n_features, out_dim, window_size, hyperparams['memory_dim'], hyperparams['num_memory_slots'], args.node_embed_dim, args.num_heads,
                        hyperparams['gru_num_layers'], hyperparams['gru_hid_dim'], args.tcn_embed_dim, args.tcn_kernel_size, hyperparams['fc_n_layers'], 
                        hyperparams['gru_hid_dim'], args.vae_latent_dim, args.recon_hid_dim, args.recon_n_layers, hyperparams['dropout'], args.alpha)
 
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
        test_loss = trainer.validate(test_loader)
        
        # Return test loss (or any metric you want to optimize)
        return test_loss[2]  # Return total test loss

param_distributions = {
    'batch_size': [32, 64, 128, 256],        
    'window_size': [50, 100, 150],           # Reasonable for short vs. medium sequences
    'n_epochs': [5, 30, 50, 100],          # Fewer than 30 often insufficient, >100 can be tested if time allows
    'init_lr': [1e-3, 3e-4, 1e-4, 1e-5],      # 1e-2 can be too big for many RNN/attention combos
    'dropout': [0.1, 0.2, 0.4, 0.5],         # 0.3 was fine, but 0.4 often helps deeper or larger models
    'memory_dim': [16, 32, 64],         # 8 might be too small to store rich patterns
    'num_memory_slots': [8, 16, 32],    # 64 can be large; you can bring it back if you have big data
    'gru_num_layers': [1, 2, 3],        # 5 might be excessive unless you have large data/time
    'gru_hid_dim': [64, 128, 150],      # 150 is your baseline, also try smaller/larger for coverage
    'fc_n_layers': [1, 2, 3]           # 5 can be quite deep; 1-3 often suffice
    # 'fc_hid_dim': [64, 128, 150]       # Similarly, vary this to see if smaller/larger FC helps
    }
# param_distributions = {
#         'batch_size': [256],        # Typically 256+ can be large for GRU-based models
#         'window_size': [100],           # Reasonable for short vs. medium sequences
#         'n_epochs': [5, 30, 50, 100],          # Fewer than 30 often insufficient, >100 can be tested if time allows
#         'init_lr': [1e-5],      # 1e-2 can be too big for many RNN/attention combos
#         'dropout': [0.4],         # 0.3 was fine, but 0.4 often helps deeper or larger models
#         'memory_dim': [16],         # 8 might be too small to store rich patterns
#         'num_memory_slots': [16],    # 64 can be large; you can bring it back if you have big data
#         'gru_num_layers': [3],        # 5 might be excessive unless you have large data/time
#         'gru_hid_dim': [150],      # 150 is your baseline, also try smaller/larger for coverage
#         'fc_n_layers': [3]           # 5 can be quite deep; 1-3 often suffice
      
#     }

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
         
        final_loss = train_and_evaluate_model(hyperparams)
        
         # 2) If NaN, penalize with a large numeric value or skip
        if torch.isnan(torch.tensor(final_loss)):
            final_loss = 1e12  # penalize heavily
            # raise ValueError("NaN encountered for this hyperparam combination")


        # 3) Store this loss on the estimator
        self._final_loss = final_loss

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


    #----------------------------------------------------------
    X_np = x_train.cpu().numpy() if isinstance(x_train, torch.Tensor) else x_train

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X_np)):
        print(f"[DEBUG] Fold {i} -> train size={len(train_idx)}, test size={len(test_idx)}")
    #----------------------------------------------------------

    grid_search = GridSearchCV(
    mtad_gat_model, 
    param_grid=param_distributions, 
    scoring='neg_mean_squared_error',  # Optimize for lower loss
    cv=kf,  # 3-fold cross-validation
    verbose=2
    )

    # Start grid search
    grid_search.fit(x_train)

    #--------------------------------------
    cv_results = grid_search.cv_results_  # dictionary of lists

    all_results = []
    for i, params in enumerate(cv_results['params']):
        mean_score = cv_results['mean_test_score'][i]
        std_score  = cv_results['std_test_score'][i]
        rank_score = cv_results['rank_test_score'][i]

        # Convert from NumPy types to standard Python types if necessary
        result_entry = {
            'params': str(params),
            'mean_test_score': float(mean_score),
            'std_test_score': float(std_score),
            'rank_test_score': int(rank_score)
        }
        all_results.append(result_entry)

    # Write to JSON
    filename = f"grid_search_results_{id}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"Grid search results saved to {filename}")

    #------------------------------------

    # Display the best parameters and score
    print("Best hyperparameters found:", grid_search.best_params_)
    print("Best score (negative loss):", grid_search.best_score_)