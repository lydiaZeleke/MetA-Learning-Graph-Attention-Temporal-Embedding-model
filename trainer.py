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

           
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #/''''''''''''''' META TRAINING AND EVALUATION '''''''''''''''\

    def compute_loss(self, inputs, targets, recons, preds):
        # Example from your code
        prediction_loss = torch.sqrt(self.pred_criterion(targets, preds))
        recon_loss = torch.sqrt(self.recon_criterion(inputs, recons))
        return prediction_loss + recon_loss

    def meta_train(self, tasks, num_epochs=5, adaptation_steps=1):
        """
        :param tasks: dict { task_id: (support_loader, query_loader), ... }
        :param num_epochs: number of meta-training epochs
        :param adaptation_steps: number of gradient updates on each support set
        """
        for epoch in range(num_epochs):
            self.model.train()
            self.outer_optimizer.zero_grad()

            # tasks = {k: tasks[k] for k in list(tasks.keys())[:1]}  #For debugging only --- limiting to a single task for quick evaluation

            # Iterate over your tasks
            for task_id, (support_loader, query_loader) in tasks.items():

                # -- 1) Create a 'functional' version of the model inside higher.innerloop_ctx --
                # We supply the *inner optimizer* (e.g. SGD).
                # For FOMAML => track_higher_grads=False (no second derivative).
                with higher.innerloop_ctx(
                        self.model, 
                        torch.optim.SGD(self.model.parameters(), lr=self.inner_lr),
                        copy_initial_weights=True,
                        track_higher_grads=False
                ) as (fmodel, diffopt):

                    # -- 2) Inner Loop: adapt on support set --
                    for _ in range(adaptation_steps):
                        # Option: sum or average loss across all support batches
                        support_loss_accum = 0.0

                        for (x_sup, y_sup) in tqdm(support_loader, desc=f"Task Specific Training- task: {task_id}"):
                            x_sup, y_sup = x_sup.to(self.device), y_sup.to(self.device)
                            recons_sup, preds_sup = fmodel(x_sup)
                            
                            ############# Dimension adjustment #####
                            if self.target_dim[0] is not None:
                                x_sup = x_sup[:, :, self.target_dim].squeeze(-1)
                                y_sup = y_sup[:, :, self.target_dim].squeeze(-1)

                            if preds_sup.ndim == 3:
                                preds_sup = preds_sup.squeeze(1)
                            if y_sup.ndim == 3:
                                y_sup = y_sup.squeeze(1)
                             ############# 

                            loss_sup = self.compute_loss(x_sup, y_sup, recons_sup, preds_sup)
                            support_loss_accum += loss_sup
                        support_loss_accum /= len(support_loader)
                        
                        # Update the functional model's params
                        diffopt.step(support_loss_accum)

                    # -- 3) Outer Loop: compute query-set loss on the adapted model --
                    query_loss_accum = 0.0
                    for (x_q, y_q) in tqdm(query_loader, desc=f"Meta-Training - task: {task_id}"):
                        x_q, y_q = x_q.to(self.device), y_q.to(self.device)
                        recons_q, preds_q = fmodel(x_q)  # fmodel has updated weights
                        ############# Dimension adjustment #####
                        if self.target_dim[0] is not None:
                            x_q = x_q[:, :, self.target_dim].squeeze(-1)
                            y_q = y_q[:, :, self.target_dim].squeeze(-1)

                        if preds_q.ndim == 3:
                            preds_q = preds_q.squeeze(1)
                        if y_q.ndim == 3:
                            y_q = y_q.squeeze(1)
                        ############# 
                        loss_q = self.compute_loss(x_q, y_q, recons_q, preds_q)
                        query_loss_accum += loss_q
                    query_loss_accum /= len(query_loader)

                    # Backprop w.r.t. the *original* meta-model (self.model)
                    query_loss_accum.backward()

            # After all tasks, update the meta-parameters
            self.outer_optimizer.step()
            print(f"[Epoch {epoch+1}/{num_epochs}] done.")

    def meta_test(self, anomaly_det_args, test_tasks, label_tasks, adaptation_steps=1):
        """
        Evaluate on new tasks.
        :param test_tasks: dict { task_id: (support_loader, query_loader), ... }
        :param y_test: ground-truth anomaly labels for the query sets
        """
        self.model.eval()
        all_results = []
        meta_mode = True

        window_size=anomaly_det_args[0]
        n_features=anomaly_det_args[1]
        pred_args=anomaly_det_args[2]

       
        for task_idx, (task_id, (support_loader, query_loader)) in enumerate(test_tasks.items()):
            # We can adapt the model again, but for meta-testing 
            # we often do real gradient steps on 'fmodel' (no second-order needed).
            with higher.innerloop_ctx(
                self.model,
                torch.optim.SGD(self.model.parameters(), lr=self.inner_lr),
                copy_initial_weights=True,
                track_higher_grads=False
            ) as (fmodel, diffopt):
                # Adapt with support set
                for _ in range(adaptation_steps):
                    support_loss_accum = 0.0
                    for (x_sup, y_sup) in support_loader:
                        x_sup, y_sup = x_sup.to(self.device), y_sup.to(self.device)
                        recons_sup, preds_sup = fmodel(x_sup)
                        ############# Dimension adjustment #####
                        if self.target_dim[0] is not None:
                            x_sup = x_sup[:, :, self.target_dim].squeeze(-1)
                            y_sup = y_sup[:, :, self.target_dim].squeeze(-1)

                        if preds_sup.ndim == 3:
                            preds_sup = preds_sup.squeeze(1)
                        if y_sup.ndim == 3:
                            y_sup = y_sup.squeeze(1)
                        ############# 
                        support_loss_accum += self.compute_loss(x_sup, y_sup, recons_sup, preds_sup)
                    support_loss_accum /= len(support_loader)
                    diffopt.step(support_loss_accum)
                
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
                    task_y_test = label_tasks[task_id][1].dataset.data

                    #remove encounter column in the label data
                    support_data = support_loader.dataset.data
                    query_data = query_loader.dataset.data
                    task_y_test = task_y_test[:,-2]
                    label_data = task_y_test[window_size:]
                    all_results = anomaly_detector.predict_anomalies(support_data, query_data, label_data, task_idx=task_id)


                    # # Evaluate
                    # epsilon_results = epsilon_eval(support_scores['A_Score_Global'],
                    #                                 query_scores['A_Score_Global'],
                    #                                 task_y_test)
                    # pot_results = pot_eval(support_scores['A_Score_Global'],
                    #                         query_scores['A_Score_Global'],
                    #                         task_y_test,
                    #                         q=..., level=...)
                    # bf_results = bf_search(query_scores['A_Score_Global'], task_y_test)

                    # print(f"Task {task_idx + 1}: Epsilon F1={epsilon_results['f1']:.4f}, "
                    #         f"POT F1={pot_results['f1']:.4f}, BF F1={bf_results['f1']:.4f}")

                    # epsilon_results = epsilon_eval(support_scores['A_Score_Global'], 
                    #                     query_scores['A_Score_Global'], 
                    #                     task_y_test, reg_level=anomaly_det_args[2]['reg_level']))
                    # pot_results = pot_eval(support_scores['A_Score_Global'], 
                    #                     query_scores['A_Score_Global'], 
                    #                     task_y_test, 
                    #                     q=anomaly_det_args[2]['q'], 
                    #                     level=anomaly_det_args[2]['level'])
                    # bf_results = bf_search(query_scores['A_Score_Global'], task_y_test)

                    # # Log results
                    # print(f"Task {task_idx + 1}:")
                    # print(f"  Epsilon Method: F1={epsilon_results['f1']:.4f}, Precision={epsilon_results['precision']:.4f}, Recall={epsilon_results['recall']:.4f}")
                    # print(f"  POT Method: F1={pot_results['f1']:.4f}, Precision={pot_results['precision']:.4f}, Recall={pot_results['recall']:.4f}")
                    # print(f"  Best-F1 Search: F1={bf_results['f1']:.4f}, Precision={bf_results['precision']:.4f}, Recall={bf_results['recall']:.4f}")


                    # all_results.append({"epsilon": epsilon_results, "pot": pot_results, "bf": bf_results})
                   

        # Aggregate final results
        return self.aggregate_results(all_results)

    def aggregate_results(self, all_results):
        # Example aggregator
        return {}
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
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc="Training")):

                assert not torch.isnan(inputs).any(), f"NaNs detected in data at epoch {epoch}, batch {batch_idx}"
                assert not torch.isnan(targets).any(), f"NaNs detected in labels at epoch {epoch}, batch {batch_idx}"
        
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs_org = inputs
                self.optimizer.zero_grad()
                # Alternative #1- GRU as the forecasting method and reconstruction output is compared to GAT's output
                # gat_struct, recons, preds = self.model(inputs)
                # #######################

                # Alternative #2- Reconstruction of GRU output and a new forecasting model

                recons, preds = self.model(inputs)
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
            for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Validation")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # targets = targets.squeeze(1)
                # Forward pass
                recons, preds = self.model(inputs)
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

