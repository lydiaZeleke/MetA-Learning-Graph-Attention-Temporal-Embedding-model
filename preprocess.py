from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from arguments import get_parser
from utils import save_tensors_to_pickle


class TSDataset(Dataset):
    def __init__(self, category, typeData, filename = None):

        if typeData == 'CUSTOM':
            dataset_folder = f"datasets/{typeData}"
            filename  = category + ".csv"
            data = pd.read_csv(path.join(dataset_folder, filename))
            
            
        elif typeData == 'SMD':
            dataset_folder = f"datasets/{typeData}"
            if category in ['train', 'test']:
                folder = category
            elif 'label' in category.lower():
                folder = 'test_label'
            else:      
                raise ValueError("SMD filename doesn't indicate train/test/label_data_test")

            data = pd.read_csv(path.join(dataset_folder, folder, filename), delimiter=',', header=None)

        elif typeData in ['MSL', 'SMAP']:
            dataset_folder = "datasets/data"
            folder = category
            data = np.load(path.join(dataset_folder, folder, filename + ".npy"))
    
        else:
            print('Wrong Dataset Argument')

        self.samples_num = data.shape[0]
        if isinstance(data, pd.DataFrame):
            x_data = data.values
        else:
            x_data = data
        self.x_data = x_data

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.samples_num


def normalize_data(data, scaler=None, normalize_technique="min_max"):

    data = np.asarray(data, dtype=np.float32)
    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)

    if scaler is None:
        if normalize_technique == "min_max":
            scaler = MinMaxScaler()
        elif normalize_technique == "z_score":
            scaler = StandardScaler()
        scaler.fit(data)

    data = scaler.transform(data)
    print("Data normalized with technique:", normalize_technique)

    return data, scaler


def preprocess_save_data(dataset, normalize, filename= None):

    x_train = TSDataset("train", dataset, filename).x_data
    x_test = TSDataset("test", dataset, filename).x_data

    if normalize:
        normalize_technique = args["normalize_technique"].lower()
        x_train, scaler = normalize_data(x_train, scaler=None, normalize_technique=normalize_technique)
        x_test, _ = normalize_data(x_test, scaler=scaler, normalize_technique=normalize_technique)

    if dataset not in ['MSL', 'SMAP']:
        y_test = TSDataset("label_data_test", dataset, filename).x_data
    else:
        y_test= None
    return x_train, x_test, y_test


def fetch_data(args):
    dataset = args["dataset"].upper()
    normalize = args["normalize"]
   
    if dataset == 'SMD':
        dataset_folder = f"datasets/{dataset}"
        output_folder = f"datasets/{dataset}/processed"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                x_train, x_test, y_test= preprocess_save_data(dataset, normalize, filename)
                filename = filename.strip('.txt') + "_"
                data_save_path = path.join(output_folder, filename + "processed.pkl")
                save_tensors_to_pickle((x_train, x_test, y_test), data_save_path)

    elif dataset == 'CUSTOM':
        output_folder = f"datasets/{dataset}/processed"
        makedirs(output_folder, exist_ok=True)
        x_train, x_test, y_test= preprocess_save_data(dataset, normalize)
        data_save_path = path.join(output_folder, "processed.pkl")
        save_tensors_to_pickle((x_train, x_test, y_test), data_save_path) 

    elif dataset in ['SMAP', 'MSL']:
        dataset_folder = "datasets/data"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True
            labels.extend(label)

        test_labels = labels
        # print(dataset, "test_label", labels.shape)

        x_train_conc = []
        x_test_conc= []
        for row in data_info:
                filename = row[0]
                x_train, x_test, _= preprocess_save_data(dataset, normalize, filename)
                x_train_conc.extend(x_train)
                x_test_conc.extend(x_test)

        # --- Use the ff three lines of code to normalize all data files together
        # normalize_technique = args["normalize_technique"].lower()
        # x_train_conc, scaler = normalize_data(x_train_conc, scaler=None, normalize_technique=normalize_technique)
        # x_test_conc, _ = normalize_data(x_test_conc, scaler=scaler, normalize_technique=normalize_technique)
        data_save_path = path.join(output_folder, dataset + "_processed.pkl")
        save_tensors_to_pickle((np.array(x_train_conc), np.array(x_test_conc), np.array(test_labels)), data_save_path)
    
    else:
        print('Wrong Dataset Argument')
    

if __name__ == "__main__":
    # Parse arguments from default and user input
    default_parser = get_parser()
    default_args = vars(default_parser.parse_args([]))  # Load default args

    user_parser = get_parser()
    user_args = vars(user_parser.parse_args())

    # Merge user-provided and default arguments
    args = {**default_args, **{k: v for k, v in user_args.items() if v is not None}}

    # Run data processing
    fetch_data(args)

    # data_save_path = path.join("datasets", args["dataset"], "processed_data.pkl")
    # save_tensors_to_pickle((x_train, x_test, y_test), data_save_path)

