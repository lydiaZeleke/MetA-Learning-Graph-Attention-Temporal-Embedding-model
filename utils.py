import torch 
# import torchvision
import numpy as np
from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from pickle import dump, load
from sklearn.model_selection import train_test_split


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window,target_dim=None,  stride = 1, horizon=1):
        # self.data = data
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert to float32
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        self.stride = stride # Overlap extent between sequences (between window sized sequences)

    def __getitem__(self, index):
        actual_index = index * self.stride
        x = self.data[actual_index : actual_index + self.window]
        y = self.data[actual_index + self.window : actual_index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


class SlidingWindowDataset2(Dataset):
    
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

        # Store sliding windows as a flat list of windows across all encounters
        self.windows = []
        self.encounter_indices = []  # Store which encounter each window belongs to

        unique_encounters = np.unique(data[:, -1])
        for encounter in unique_encounters:
            encounter_data = data[data[:, -1] == encounter]
            num_windows = len(encounter_data) - window

            # Create sliding windows for each encounter
            for i in range(num_windows):
                x = encounter_data[i : i + window]
                y = encounter_data[i + window : i + window + horizon]
                self.windows.append((x, y))
                self.encounter_indices.append(encounter)  # Track encounter for each window

    def __getitem__(self, index):
        # Return the specific sliding window (x, y) pair for the given index
        return self.windows[index]

    def __len__(self):
        # Return the total number of sliding windows across all encounters
        return len(self.windows)

    def get_encounter_indices(self):
        # Return the list of encounter indices for each window
        return self.encounter_indices

# Save tensors as NumPy arrays or raw data
def save_tensors_to_pickle(tensors, file_path):
    tensors = [tensor.numpy() if hasattr(tensor, "numpy") else tensor for tensor in tensors]
    with open(file_path, 'wb') as f:
        dump(tensors, f)
    print(f"Tensors saved to {file_path}")

# Reload data and wrap it back into required format if needed
def load_tensors_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        tensors = load(f)
    print(f"Tensors loaded from {file_path}")
    return [torch.from_numpy(tensor) if isinstance(tensor, np.ndarray) else tensor for tensor in tensors]


def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP":
        return None #[0]
    elif dataset == "MSL":
        return None #[0]
    elif dataset == "CUSTOM":
        return None
    elif dataset == "SMD":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))

def generate_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    dataset_size = len(train_dataset)
    split = int(np.floor(val_split * dataset_size))
    
    if val_split > 0.0:
        # Contiguous split for time-series data
        train_indices, val_indices = list(range(dataset_size - split)), list(range(dataset_size - split, dataset_size))
        train_sampler = torch.utils.data.SequentialSampler(train_indices)
        val_sampler = torch.utils.data.SequentialSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        print(f"train_size: {dataset_size}")

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def adjust_shape(data, window_size, is_score=False, test_data=None):
    reshaped_data = []
    
    if is_score and test_data is not None:
        # Get unique encounters from test_data
        unique_encounters = np.unique(test_data[:, -1])
        
        # Initialize an index tracker to manage the start of each encounter in `data`
        current_start_index = 0
        
        for encounter in unique_encounters:
            # Find the corresponding data for the current encounter in test_data
            encounter_data = test_data[test_data[:, -1] == encounter]
            
            # Calculate the length of the current encounter
            encounter_length = len(encounter_data)
            
            # Calculate the start and end indices for the anomaly score (data)
            start_index = current_start_index + window_size  # Skip first `window_size` points
            end_index = current_start_index + encounter_length  # Full length of the encounter
            
            # Append the corresponding segment of the anomaly score
            reshaped_data.append(data[start_index:end_index])
            
            # Move the start index for the next encounter
            current_start_index += encounter_length
    
    else:
        # Reshaping regular data (not score) based on encounter values in the data
        unique_encounters = np.unique(data[:, -1])
        for encounter in unique_encounters:
            # Extract data for this encounter
            encounter_data = data[data[:, -1] == encounter]
            
            # Trim first `window_size` data points from each encounter
            trimmed_encounter = encounter_data[window_size:]  # Remove first `window_size` points
            reshaped_data.append(trimmed_encounter)

    # Combine back into one array
    reshaped_data = np.concatenate(reshaped_data, axis=0)
    return reshaped_data


#'''''''''''''''''''  MAML Implementation ''''''''''''''''''#
# Function to split dataset into tasks
def split_dataset_into_tasks(data):
    tasks = {}
    encounter_numbers = np.unique(data[:, -1])
    for encounter_num in encounter_numbers:
        task_data = data[data[:, -1] == encounter_num]
        tasks[int(encounter_num)] = task_data
    return tasks

# Function to split each task into support and query sets
def split_task_into_support_and_query(task_data, test_size=0.2):
    support_data, query_data = train_test_split(task_data, test_size=test_size, shuffle=True, random_state=42)
    return support_data, query_data

# Generate DataLoaders for support and query sets
def generate_task_dataloaders(tasks, window_size, target_dims, batch_size):
    task_dataloaders = {}
    for task_id, task_data in tasks.items():
        support_data, query_data = split_task_into_support_and_query(task_data)

        support_dataset = SlidingWindowDataset(support_data, window_size, target_dims)
        query_dataset = SlidingWindowDataset(query_data, window_size, target_dims)

        support_loader = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)
        query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)

        task_dataloaders[task_id] = (support_loader, query_loader)

    return task_dataloaders
def merge_labels_w_encs (x_test, y_test): 
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    encounter_col = x_test[:, -1].reshape(-1, 1)     # shape (N,1)

    merged_y_test = np.hstack([y_test, encounter_col])  # shape (N, D)
    return merged_y_test

# Main function to process x_train and x_test
def prepare_dataloaders(x_train, x_test, y_test, window_size, target_dims, batch_size):
    # Split datasets into tasks based on encounter number
    train_tasks = split_dataset_into_tasks(x_train)
    test_tasks = split_dataset_into_tasks(x_test)

    y_test = merge_labels_w_encs(x_test, y_test)
    label_tasks = split_dataset_into_tasks(y_test)


    # Generate DataLoaders for training and testing tasks
    train_dataloaders = generate_task_dataloaders(train_tasks, window_size, target_dims, batch_size)
    test_dataloaders = generate_task_dataloaders(test_tasks, window_size, target_dims, batch_size)
    label_dataloaders = generate_task_dataloaders(label_tasks, window_size, target_dims, batch_size)



    return train_dataloaders, test_dataloaders, label_dataloaders
