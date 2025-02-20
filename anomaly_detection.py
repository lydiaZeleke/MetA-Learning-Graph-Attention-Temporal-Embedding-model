import json
from tqdm import tqdm
from library.eval_methods import *
from utils import *
import pandas as pd


class AnomalyDetector:
    """  ****** Class taken from MTAD-GAT git implementation  (https://github.com/ML4ITS/mtad-gat-pytorch) *******

    :param model: model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, window_size, n_features, pred_args, meta_learning_mode = False, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name
        self.meta_mode = meta_learning_mode

    def get_score(self, values):
        """Calculate anomaly scores using the hybrid model (GRU + TCN + GAT).
        :param values: 2D array of multivariate time-series data, shape (N, num_features)
        :return: DataFrame with anomaly scores and predictions for each feature
        """
        print("Predicting and calculating anomaly scores...")
        
        # Create sliding window dataset and data loader
        data = SlidingWindowDataset(values, self.window_size)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        
        self.model.eval()
        reconstructions, forecasts = [], []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                # Forward pass through the model
                reconstruction_output, forecast_output = self.model(x)


                # Shift input to include observed value (y) for reconstruction
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)  # Sliding reconstruction
                recon_window, _ = self.model(recon_x)

                reconstructions.append(recon_window[:, -1, :].cpu().numpy())  # Last reconstruction
                forecasts.append(forecast_output.cpu().numpy())  # Forecast predictions

        # Convert outputs to numpy arrays
        reconstructions = np.concatenate(reconstructions, axis=0)  # (N, num_features)
        forecasts = np.concatenate(forecasts, axis=0)  # (N, num_features)
        actual = values[self.window_size:]  # Actual values (N, num_features)
        
        actual = actual.numpy()


        # Initialize anomaly score storage
        anomaly_scores = np.zeros_like(actual)
        df_dict = {}

        # Feature-wise anomaly scores
        for i in range(forecasts.shape[1]):
            df_dict[f"Forecast_{i}"] = forecasts[:, i]
            df_dict[f"Recon_{i}"] = reconstructions[:, i]
            df_dict[f"True_{i}"] = actual[:, i]

            # Anomaly score: Combine forecast and reconstruction errors
            forecast_error = (forecasts[:, i] - actual[:, i]) ** 2
            recon_error = (reconstructions[:, i] - actual[:, i]) ** 2
            a_score = np.sqrt(forecast_error) + self.gamma * np.sqrt(recon_error)

            # Optionally scale scores
            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1 + iqr)

            anomaly_scores[:, i] = a_score
            df_dict[f"A_Score_{i}"] = a_score

        # Aggregate anomaly scores globally
        anomaly_scores_global = np.mean(anomaly_scores, axis=1)
        df_dict['A_Score_Global'] = anomaly_scores_global

        return pd.DataFrame(df_dict)

    def predict_anomalies(self, train, test, true_anomalies, load_scores=False, save_output=True,
                          scale_scores=False, task_idx= None):
        """ Predicts anomalies

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        :param scale_scores: Whether to feature-wise scale anomaly scores
        """

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            train_pred_df = self.get_score(train)
            test_pred_df = self.get_score(test)

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

            if not self.meta_mode:
                train_anomaly_scores = self.adjust_anomaly_scores(train_anomaly_scores, self.dataset, True, self.window_size)
                test_anomaly_scores = self.adjust_anomaly_scores(test_anomaly_scores, self.dataset, False, self.window_size)

            # Update df
            train_pred_df['A_Score_Global'] = train_anomaly_scores
            test_pred_df['A_Score_Global'] = test_anomaly_scores

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        # Find threshold and predict anomalies at feature-level (for plotting and diagnosis purposes)
        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))
        for i in range(out_dim):
            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
            epsilon = find_epsilon(train_feature_anom_scores, reg_level=2)

            train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(int)
            test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)

            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds

            train_pred_df[f"Thresh_{i}"] = epsilon
            test_pred_df[f"Thresh_{i}"] = epsilon

            all_preds[:, i] = test_feature_anom_preds

        # Global anomalies (entity-level) are predicted using aggregation of anomaly scores across all features
        # These predictions are used to evaluate performance, as true anomalies are labeled at entity-level
        # Evaluate using different threshold methods: brute-force, epsilon and peaks-over-treshold
        e_eval = epsilon_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies, reg_level=self.reg_level)
        p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,
                          q=self.q, level=self.level, dynamic=self.dynamic_pot)
        if true_anomalies is not None:
            bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0.01, end=2, step_num=100, verbose=False)
        else:
            bf_eval = {}

        if self.meta_mode:
             print(f"Task {task_idx + 1}:")
        print(f"Results using epsilon method:\n {e_eval}")
        print(f"Results using peak-over-threshold method:\n {p_eval}")
        print(f"Results using best f1 score search:\n {bf_eval}")

        for k, v in e_eval.items():
            if not type(e_eval[k]) == list:
                e_eval[k] = float(v)
        for k, v in p_eval.items():
            if not type(p_eval[k]) == list:
                p_eval[k] = float(v)
        for k, v in bf_eval.items():
            bf_eval[k] = float(v)

        # Save
        summary = {"epsilon_result": e_eval, "pot_result": p_eval, "bf_result": bf_eval}

        if self.meta_mode:
            return summary

        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        # Save anomaly predictions made using epsilon method (could be changed to pot or bf-method)
        if save_output:
            global_epsilon = e_eval["threshold"]
            test_pred_df["A_True_Global"] = true_anomalies
            train_pred_df["Thresh_Global"] = global_epsilon
            test_pred_df["Thresh_Global"] = global_epsilon
            train_pred_df[f"A_Pred_Global"] = (train_anomaly_scores >= global_epsilon).astype(int)
            test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
            # Adjust predictions according to evaluation strategy
            if true_anomalies is not None:
                test_preds_global = adjust_predicts(None, true_anomalies, global_epsilon, pred=test_preds_global)
            test_pred_df[f"A_Pred_Global"] = test_preds_global

            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("-- Done.")

    def adjust_anomaly_scores(self, scores, dataset, is_train, lookback):
        """
        Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
        :param scores: anomaly_scores
        :param dataset: name of dataset
        :param is_train: if scores is from train set
        :param lookback: lookback (window size) used in model
        """
        # Initialize adjusted_scores as a copy of the original scores
        adjusted_scores = scores.copy()

        # Remove errors for time steps when transitioning to a new channel/encounter
        if dataset.upper() not in ['SMAP', 'MSL', 'CUSTOM']:
            return adjusted_scores
        
        # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
        if dataset.upper() == 'CUSTOM':
            # Identify transition points for CUSTOM dataset based on enc_num changes
            if is_train:
                md = pd.read_csv(f'./datasets/{dataset.upper()}/train.csv')
            else:
                md = pd.read_csv(f'./datasets/{dataset.upper()}/test.csv')
            md_columns = md.columns
            md = md.to_numpy()  
            md = adjust_shape(md, lookback)
            md = pd.DataFrame(md, columns= md_columns)  
            sep_cuma = np.where(np.diff(md['enc_num']) != 0)[0] + 1 - lookback
            sep_cuma = sep_cuma[sep_cuma > 0]

            # Apply adjustments around encounter boundaries
            buffer = np.arange(1, 20)
            i_remov = np.sort(np.concatenate((sep_cuma, np.array([i + buffer for i in sep_cuma]).flatten(),
                                            np.array([i - buffer for i in sep_cuma]).flatten())))
            i_remov = i_remov[(i_remov < len(scores)) & (i_remov >= 0)]
            adjusted_scores[i_remov] = 0

            # Normalize each encounter's scores individually
            s = [0] + sep_cuma.tolist() + [len(scores)]
            for c_start, c_end in zip(s[:-1], s[1:]):
                e_s = adjusted_scores[c_start: c_end]
                # e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s) + 1e-8)  # Avoid division by zero
                e_s = (e_s + 1e-5 - np.min(e_s)) / (np.max(e_s) - np.min(e_s) + 1e-8)

                adjusted_scores[c_start: c_end] = e_s
    
        else:
            adjusted_scores = scores.copy()
            if is_train:
                md = pd.read_csv(f'./datasets/data/{dataset.lower()}_train_md.csv')
            else:
                md = pd.read_csv('./datasets/data/labeled_anomalies.csv')
                md = md[md['spacecraft'] == dataset.upper()]

            md = md[md['chan_id'] != 'P-2']

            # Sort values by channel
            md = md.sort_values(by=['chan_id'])

            # Getting the cumulative start index for each channel
            sep_cuma = np.cumsum(md['num_values'].values) - lookback
            sep_cuma = sep_cuma[:-1]
            buffer = np.arange(1, 20)
            i_remov = np.sort(np.concatenate((sep_cuma, np.array([i+buffer for i in sep_cuma]).flatten(),
                                            np.array([i-buffer for i in sep_cuma]).flatten())))
            i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
            i_remov = np.sort(np.unique(i_remov))
            if len(i_remov) != 0:
                adjusted_scores[i_remov] = 0

            # Normalize each concatenated part individually
            sep_cuma = np.cumsum(md['num_values'].values) - lookback
            s = [0] + sep_cuma.tolist()
            for c_start, c_end in [(s[i], s[i+1]) for i in range(len(s)-1)]:
                e_s = adjusted_scores[c_start: c_end+1]

                e_s = (e_s - np.min(e_s))/(np.max(e_s) - np.min(e_s))
                adjusted_scores[c_start: c_end+1] = e_s

        return adjusted_scores