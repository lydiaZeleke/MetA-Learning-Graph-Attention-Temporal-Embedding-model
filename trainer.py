import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from os import path, makedirs
import time
from datetime import datetime 
import matplotlib.pyplot as plt
import higher
from copy import deepcopy
from anomaly_detection import AnomalyDetector
from library.eval_methods import *

torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, target_dim, model, train_loader, val_loader, criterion, optimizer, device, save_path, checkpoint_dir="checkpoints", inner_lr=0.01, meta_lr= 0.001):
        """
        Initialize the trainer with model, data loaders, loss, optimizer, and device.
        :param model: PyTorch model to train
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param criterion: Loss function
        :param optimizer: Optimizer
        :param device: Device ('cpu' or 'cuda')
        :param checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.target_dim = target_dim,
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pred_criterion = criterion
        self.recon_criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.checkpoint_dir = checkpoint_dir
        self.print_every=1
        self.losses = {
            "train_prediction": [],
            "train_reconstruction": [],
            "train_combined": [],
            "val_prediction": [],
            "val_reconstruction":[],
            "val_combined":[]
        }
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr

        self.epoch_times = []
        makedirs(checkpoint_dir, exist_ok=True)
                
        # Outer (meta) optimizer on self.model
        self.outer_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.model.to(self.device)
        # self.model = torch.compile(self.model)

           
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    #/''''''''''''''' META TRAINING AND EVALUATION '''''''''''''''\

    def generate_json_log_writer(self, base_dir="meta_training_logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        support_log_path = f"{base_dir}/support_logs_{timestamp}.json"
        query_log_path = f"{base_dir}/query_logs_{timestamp}.json"

        def write_logs(support_logs, query_logs):
            with open(support_log_path, "w") as f:
                json.dump(support_logs, f, indent=4)
            with open(query_log_path, "w") as f:
                json.dump(query_logs, f, indent=4)
            return support_log_path, query_log_path

        return write_logs
    
    def generate_csv_log_writer(self, base_dir="meta_training_logs", is_meta_train=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_type= timestamp + "_train" if is_meta_train else timestamp + "_test"
        support_log_path = f"{base_dir}/support_logs_{timestamp_type}.csv"
        query_log_path = f"{base_dir}/query_logs_{timestamp_type}.csv"

        def write_logs_to_csv(support_logs, query_logs):
            with open(support_log_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["task_id", "pred_loss", "recon_loss", "total_loss"])
                writer.writeheader()
                writer.writerows(support_logs)

            with open(query_log_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["task_id", "pred_loss", "recon_loss", "total_loss"])
                writer.writeheader()
                writer.writerows(query_logs)

            return support_log_path, query_log_path

        return write_logs_to_csv

    def compute_loss(self, inputs, targets, recons, preds):
        # Example from your code
        prediction_loss = torch.sqrt(self.pred_criterion(targets, preds))
        recon_loss = torch.sqrt(self.recon_criterion(inputs, recons))
        return prediction_loss, recon_loss, prediction_loss + 0.8 * recon_loss

    def meta_train(self, tasks, num_epochs=5, adaptation_steps=1):
        epoch_logs = {"support": [], "query": []}
        write_logs = self.generate_csv_log_writer(is_meta_train=True)

        # task_items = list(tasks.items())
        # random.shuffle(task_items)

        for epoch in range(num_epochs):
            self.model.train()
            self.outer_optimizer.zero_grad()
            support_losses = []
            query_losses = []
            meta_loss = 0.0

            # tasks = {k: tasks[k] for k in list(tasks.keys())[:1]}  #For debugging only --- limiting to a single task for quick evaluation
            for task_id, (support_loader, query_loader) in tasks.items():
                with higher.innerloop_ctx(
                    self.model,
                    torch.optim.SGD(self.model.parameters(), lr=self.inner_lr),
                    copy_initial_weights=True,
                    track_higher_grads=False
                ) as (fmodel, diffopt):

                    is_new_task = True
                    # ---- Inner Loop ----
                    support_loss_accum = support_pred_loss_accum = support_recon_loss_accum = 0.0
                    for _ in range(adaptation_steps):
                        for (x_sup, y_sup) in tqdm(support_loader, desc=f"Task-Specific Training - task: {task_id}"):

                            x_sup, y_sup = x_sup[:, :, :-1].to(self.device), y_sup[:, :, :-1].to(self.device)
                            recons_sup, preds_sup = fmodel(x_sup, is_new_task)
                            is_new_task = False

                            # Dimension adjustment
                            if self.target_dim[0] is not None:
                                x_sup = x_sup[:, :, self.target_dim].squeeze(-1)
                                y_sup = y_sup[:, :, self.target_dim].squeeze(-1)
                            if preds_sup.ndim == 3: preds_sup = preds_sup.squeeze(1)
                            if y_sup.ndim == 3: y_sup = y_sup.squeeze(1)

                            pred_loss_sup, recon_loss_sup, loss_sup = self.compute_loss(x_sup, y_sup, recons_sup, preds_sup)
                            support_loss_accum += loss_sup
                            support_pred_loss_accum += pred_loss_sup
                            support_recon_loss_accum += recon_loss_sup

                        support_loss_accum /= len(support_loader)
                        support_pred_loss_accum /= len(support_loader)
                        support_recon_loss_accum /= len(support_loader)
                        diffopt.step(support_loss_accum)

                    # ---- Outer Loop (Query) ----
                    query_loss_accum = query_pred_loss_accum = query_recon_loss_accum = 0.0
                    for (x_q, y_q) in tqdm(query_loader, desc=f"Meta-Training - task: {task_id}"):
                        x_q, y_q = x_q[:, :, :-1].to(self.device), y_q[:, :, :-1].to(self.device)
                        recons_q, preds_q = fmodel(x_q)
                        if self.target_dim[0] is not None:
                            x_q = x_q[:, :, self.target_dim].squeeze(-1)
                            y_q = y_q[:, :, self.target_dim].squeeze(-1)
                        if preds_q.ndim == 3: preds_q = preds_q.squeeze(1)
                        if y_q.ndim == 3: y_q = y_q.squeeze(1)

                        pred_loss_q, recon_loss_q, loss_q = self.compute_loss(x_q, y_q, recons_q, preds_q)
                        query_loss_accum += loss_q
                        query_pred_loss_accum += pred_loss_q
                        query_recon_loss_accum += recon_loss_q

                    query_loss_accum /= len(query_loader)
                    query_pred_loss_accum /= len(query_loader)
                    query_recon_loss_accum /= len(query_loader)

                    # Accumulate meta loss
                    meta_loss += query_loss_accum

                    # Logging
                    support_losses.append(support_loss_accum.item())
                    query_losses.append(query_loss_accum.item())

                    epoch_logs["support"].append({
                        "task_id": task_id,
                        "pred_loss": support_pred_loss_accum.item(),
                        "recon_loss": support_recon_loss_accum.item(),
                        "total_loss": support_loss_accum.item()
                    })

                    epoch_logs["query"].append({
                        "task_id": task_id,
                        "pred_loss": query_pred_loss_accum.item(),
                        "recon_loss": query_recon_loss_accum.item(),
                        "total_loss": query_loss_accum.item()
                    })

            # Single backward after all tasks
            meta_loss.backward()
            self.outer_optimizer.step()

            print(f"[Epoch {epoch+1}/{num_epochs}] done.")

        write_logs(epoch_logs["support"], epoch_logs["query"])

    def meta_test(self, anomaly_det_args, test_tasks, label_tasks, adaptation_steps=1):
        """
        Evaluate on new tasks.
        :param test_tasks: dict { task_id: (support_loader, query_loader), ... }
        :param y_test: ground-truth anomaly labels for the query sets
        """
        self.model.eval()
        all_results = []
        meta_mode = True
        label_index = 1

        window_size=anomaly_det_args[0]
        n_features=anomaly_det_args[1]
        pred_args=anomaly_det_args[2]

        anomaly_counts = self.count_anomalies_per_task(test_tasks, label_tasks)
        print(anomaly_counts)

        test_loss_logs = {"support": [], "query": []}
        write_logs = self.generate_csv_log_writer()
        
        for task_idx, (task_id, (support_loader, query_loader)) in enumerate(test_tasks.items()):
            # We can adapt the model again, but for meta-testing 
            # we often do real gradient steps on 'fmodel' (no second-order needed).
            is_new_task = True    
            if task_id in [13, 19, 23, 27, 29, 36, 66, 71, 82, 87]: #29
                temp_loader = support_loader
                support_loader = query_loader
                query_loader = temp_loader
                label_index = 0
            else:
                label_index = 1
            with higher.innerloop_ctx(
                self.model,
                torch.optim.SGD(self.model.parameters(), lr=self.inner_lr),
                copy_initial_weights=True,
                track_higher_grads=False
            ) as (fmodel, diffopt):
                # Adapt with support set
                for _ in range(adaptation_steps):
                    support_loss_accum = 0.0
                    support_loss_accum = support_pred_loss_accum = support_recon_loss_accum = 0.0
                    for (x_sup, y_sup) in support_loader:
                        x_sup, y_sup = x_sup[:, :, :-1].to(self.device), y_sup[:, :, :-1].to(self.device)
                        recons_sup, preds_sup = fmodel(x_sup, is_new_task)
                        is_new_task = False
                        ############# Dimension adjustment #####
                        if self.target_dim[0] is not None:
                            x_sup = x_sup[:, :, self.target_dim].squeeze(-1)
                            y_sup = y_sup[:, :, self.target_dim].squeeze(-1)

                        if preds_sup.ndim == 3:
                            preds_sup = preds_sup.squeeze(1)
                        if y_sup.ndim == 3:
                            y_sup = y_sup.squeeze(1)
                        ############# 
                    pred_loss, recon_loss, support_loss_test = self.compute_loss(x_sup, y_sup, recons_sup, preds_sup)
                    support_loss_accum += support_loss_test
                    support_loss_accum /= len(support_loader)
                    support_pred_loss_accum += pred_loss
                    support_recon_loss_accum += recon_loss
                    support_pred_loss_accum/= len(support_loader)
                    support_recon_loss_accum/= len(support_loader)
                    diffopt.step(support_loss_accum)
                
                # Logging
                    
                test_loss_logs["support"].append({
                    "task_id": task_id,
                    "pred_loss": support_pred_loss_accum.item(),
                    "recon_loss": support_recon_loss_accum.item(),
                    "total_loss": support_loss_accum.item()
                })

                test_loss_logs["query"].append({
                        "task_id": task_id,
                        "pred_loss": 0,
                        "recon_loss": 0,
                        "total_loss": 0
                })

                # Now fmodel is adapted. We can do anomaly detection or just measure performance:
                # Example: pass fmodel to your AnomalyDetector
                with torch.no_grad():
                    def adapted_forward(x_batch):
                        return fmodel(x_batch)  # recons, preds

                    
                    anomaly_detector = AnomalyDetector(fmodel, #adapted_forward,
                                                        window_size,
                                                        n_features,
                                                        pred_args,
                                                        meta_mode)
                
            
                    # Evaluate scores
                    # support_data = support_loader.dataset.data
                    # support_scores = anomaly_detector.get_score(support_data)

                    # query_data = query_loader.dataset.data
                    # query_scores = anomaly_detector.get_score(query_data)

                    # Extract the correct portion of y_test
                    
                    task_y_test = label_tasks[task_id][label_index].dataset.data

                    #remove encounter column in the label data
                    support_data = support_loader.dataset.data[:,:-1] 
                    query_data = query_loader.dataset.data[:,:-1]
                    task_y_test = task_y_test[:,-2]
                    label_data = task_y_test[window_size:]
                     # Get anomalies for this task
                 
                    task_results = anomaly_detector.predict_anomalies(support_data, query_data, label_data, task_idx=task_id)
                    all_results.append(task_results)

        write_logs(test_loss_logs["support"], test_loss_logs["query"])

        time_str = datetime.now().strftime("%d%m%Y_%H%M%S") 
        meta_eval_dir = 'meta_evaluation_results'
        filename = f"{meta_eval_dir}/results_{time_str}.json"          
        with open(filename, "w") as f:
            json.dump(all_results, f, indent=4)
        # Aggregate final results
        return self.aggregate_results(all_results)

    def aggregate_results(self, all_results):
        # Example aggregator
        return {}
    
    def count_anomalies_per_task(self, test_tasks, label_tasks):
        """
        Count the number of anomalies in support and query sets for each task.

        :param test_tasks: dict {task_id: (support_loader, query_loader), ...}
        :param label_tasks: dict {task_id: (support_loader, query_loader), ...} with anomaly labels
        :return: dict of anomaly counts per task {task_id: {'support': count, 'query': count}}
        """
        anomaly_counts = {}

        for task_id in test_tasks:
            support_anomalies = query_anomalies = 0
            support_labels = label_tasks[task_id][0].dataset.data[:, -2]  # assuming labels are in the second-to-last column
            query_labels = label_tasks[task_id][1].dataset.data[:, -2]

            support_anomalies = (support_labels == 1).sum().item()
            query_anomalies = (query_labels == 1).sum().item()

            anomaly_counts[task_id] = {
                "support_anomalies": support_anomalies,
                "query_anomalies": query_anomalies
            }

        return anomaly_counts


    #/'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''\
    def train(self, num_epochs):
        """""
        Train the model for a specified number of epochs.
        """
        self.model.to(self.device)

        # Example usage:
        total_params = self.count_parameters(self.model)
        print(f"Total Trainable Parameters: {total_params}")

        train_start = time.time()

        val_losses =[]

        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"Epoch {epoch + 1}/{num_epochs}")
            self.model.train()  # Set model to training mode

            epoch_loss = 0
            prediction_train_loss = []
            reconstruction_train_loss = []
            combined_train_loss = []
            for batch_idx, (inputs, targets, is_new) in enumerate(tqdm(self.train_loader, desc="Training")):

                assert not torch.isnan(inputs).any(), f"NaNs detected in data at epoch {epoch}, batch {batch_idx}"
                assert not torch.isnan(targets).any(), f"NaNs detected in labels at epoch {epoch}, batch {batch_idx}"
        
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                is_new   = is_new.to(self.device)        # shape (B,)

                inputs_org = inputs
                self.optimizer.zero_grad()
                # Alternative #1- GRU as the forecasting method and reconstruction output is compared to GAT's output
                # gat_struct, recons, preds = self.model(inputs)
                # #######################

                # Alternative #2- Reconstruction of GRU output and a new forecasting model

                recons, preds = self.model(inputs, is_new=is_new)
                if self.target_dim[0] is not None:
                    inputs = inputs[:, :, self.target_dim].squeeze(-1)
                    targets = targets[:, :, self.target_dim].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if targets.ndim == 3:
                    targets = targets.squeeze(1)
                ###########################
               
                prediction_loss = torch.sqrt(self.pred_criterion(targets, preds))
                recon_loss = torch.sqrt(self.recon_criterion(inputs, recons))
                loss = prediction_loss + recon_loss

                # if torch.isnan(prediction_loss) or torch.isnan(recon_loss):
                #     print(f'batch indx for NaN loss: {batch_idx}')
                # Backward pass and optimization
               
                loss.backward()
                #Gradient clipping to avoid gradient explosion

                #UNCOMMENT FOR CUSTOM
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                # loss = self.criterion(preds, targets)
                prediction_train_loss.append(prediction_loss.item())
                reconstruction_train_loss.append(recon_loss.item())

            prediction_train_loss = np.array(prediction_train_loss)
            prediction_epoch_train_loss = np.sqrt((prediction_train_loss ** 2).mean())
            self.losses["train_prediction"].append(prediction_epoch_train_loss)

            reconstruction_train_loss = np.array(reconstruction_train_loss)
            reconstruction_epoch_train_loss = np.sqrt((reconstruction_train_loss ** 2).mean())
            self.losses["train_reconstruction"].append(reconstruction_epoch_train_loss)

            combined_epoch_train_loss = reconstruction_epoch_train_loss + prediction_epoch_train_loss
            self.losses["train_combined"].append(combined_epoch_train_loss)


            if self.val_loader is not None:
                # Validate after each epoch
                prediction_epoch_val_loss, recon_epoch_val_loss,  combined_epoch_val_loss = self.validate()
                val_losses.append(combined_epoch_val_loss)
                if combined_epoch_val_loss <= self.losses["val_combined"][-1]:
                    self.save(f"model.pt")
                # print(f"Validation Loss: {val_loss:.4f}")
            else: 
                self.save(f"model.pt")

            # Early Stopping Check
            # if self.early_stopping(val_losses, patience=patience, min_delta=min_delta):
            #     print(f'Early Stopping triggered at epoch {epoch+1}') 

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            self.epoch_loss_print(epoch, epoch_time)


            # Save checkpoint
            # self.save_checkpoint(epoch)
        self.plot_loss_curves(num_epochs)     

    def validate(self, loader=None):
        """
        Validate the model on the validation dataset.
        """
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0
        prediction_val_loss = []
        recon_val_loss= []

        if loader is None:
            loader = self.val_loader
         
        with torch.no_grad():
            for batch_idx, (inputs, targets, is_new) in enumerate(tqdm(loader, desc="Validation")):
                inputs, targets, is_new = inputs.to(self.device), targets.to(self.device),is_new.to(self.device)
                # targets = targets.squeeze(1)
                # Forward pass
                recons, preds = self.model(inputs, is_new)
                if self.target_dim[0] is not None:
                    inputs = inputs[:, :, self.target_dim].squeeze(-1)
                    targets = targets[:, :, self.target_dim].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if targets.ndim == 3:
                    targets = targets.squeeze(1)
                ###########################

                prediction_loss = torch.sqrt(self.pred_criterion(targets, preds))
                recon_loss = torch.sqrt(self.recon_criterion(inputs, recons))

                # gat_struct, recons, preds = self.model(inputs)
                # prediction_loss = self.pred_criterion(targets, preds)
                # recon_loss = self.recon_criterion(gat_struct, recons)

                prediction_val_loss.append(prediction_loss.item())
                recon_val_loss.append(recon_loss.item())
                # val_loss += loss.item()

        prediction_val_loss = np.array(prediction_val_loss)
        prediction_epoch_val_loss = np.sqrt((prediction_val_loss ** 2).mean())
        self.losses["val_prediction"].append(prediction_epoch_val_loss)

        recon_val_loss = np.array(recon_val_loss)
        recon_epoch_val_loss = np.sqrt((recon_val_loss ** 2).mean())
        self.losses["val_reconstruction"].append(recon_epoch_val_loss)

        combined_epoch_val_loss = prediction_epoch_val_loss + recon_epoch_val_loss
        self.losses["val_combined"].append(combined_epoch_val_loss)
  
        # # avg_loss = val_loss / len(self.val_loader)
        return prediction_epoch_val_loss, recon_epoch_val_loss, combined_epoch_val_loss
    
    def early_stopping(self, val_losses, patience=5, min_delta=0.001):
        """
        Early stopping function to monitor validation loss.

        Args:
            val_losses (list): List of validation losses from previous epochs.
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in validation loss to qualify as an improvement.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if len(val_losses) < patience:
            return False  # Not enough epochs to compare yet

        # Check if the validation loss improved over the last `patience` epochs
        recent_losses = val_losses[-patience:]
        best_loss = min(recent_losses)

        # If no improvement greater than min_delta, trigger early stopping
        if all((loss > best_loss - min_delta) for loss in recent_losses):
            return True
        return False

    def epoch_loss_print(self, epoch, epoch_time):
        if epoch % self.print_every == 0:
            s = (
                f"[Epoch {epoch + 1}] "
                f"Train forecast loss = {self.losses['train_prediction'][epoch]:.5f}, "
                f"Train reconstruction loss = {self.losses['train_reconstruction'][epoch]:.5f}, "
                f"Train total loss = {self.losses['train_combined'][epoch]:.5f}, "
            )

            if self.val_loader is not None:
                s += (
                    f" ------val forecast loss = {self.losses['val_prediction'][epoch]:.5f}, "
                    f"Validation reconstruction loss = {self.losses['val_reconstruction'][epoch]:.5f}, "
                    f"Validation total loss = {self.losses['val_combined'][epoch]:.5f}, "
                )

            s += f" [{epoch_time:.1f}s]"
            print(s)


    def plot_loss_curves(self, num_epochs):
        """Plot training and validation loss curves over epochs."""
        epochs = range(1, len(self.losses["train_prediction"]) + 1)

        plt.figure(figsize=(12, 5))

        # Plot forecasting losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.losses["train_prediction"], label="Train Forecast Loss")
        if len(self.losses["val_prediction"]) == len(epochs):
            plt.plot(epochs, self.losses["val_prediction"], label="Val Forecast Loss")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Forecasting Loss")
        plt.legend()

        # Plot reconstruction losses
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.losses["train_reconstruction"], label="Train Recon Loss")
        if len(self.losses["val_reconstruction"]) == len(epochs):
            plt.plot(epochs, self.losses["val_reconstruction"], label="Val Recon Loss")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Reconstruction Loss")
        plt.legend()

        # Gather hyperparameters from the model for a suptitle or filename
        # Adjust based on where/how you store them. For demonstration:
        hp = []
        hyperparam_summary = self.model.get_hyperparam_summary()
        hp.append(hyperparam_summary)
        opt_group = self.optimizer.param_groups[0]
        lr = opt_group.get('lr', None)
        hp.append(f"Init lr= {lr}")
        hp.append(f"Epochs={num_epochs}")
        hyperparam_summary = ", ".join(hp)
        plt.suptitle(hyperparam_summary, fontsize=10)
     

        plt.tight_layout()

        # Draw the figure so it won't block execution, then briefly pause to let it render
        plt.draw()
        plt.pause(0.001)

        # Save figure in subfolder "plots"
        makedirs("plots", exist_ok=True)
        # Generate a timestamped filename, e.g. "loss_curve_20230805_132045.png"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"plots/loss_curve_{timestamp}.png"

        # Use bbox_inches="tight" so the suptitle doesn't get cut off.
        plt.savefig(filename, bbox_inches="tight")
        print(f"Plot saved to {filename}")


    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.save_path + "/" + file_name
        if path.exists(self.save_path):
            pass
        else:
            makedirs(self.save_path)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def save_checkpoint(self, epoch):
        """
        Save the model checkpoint.
        """
        checkpoint_path = path.join(self.checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint.
        """
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print(f"Model checkpoint loaded from {checkpoint_path}")

