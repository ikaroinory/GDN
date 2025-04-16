import copy
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import TimeDataset
from models import GDN
from . import Arguments, Logger
from .evaluate import get_metrics


class Runner:
    def __init__(self):
        self.args = Arguments()

        # Initialize
        self.__set_seed()

        self.num_nodes: int = -1
        self.model_save_path = f'saves/{self.args.dataset}'

        self.start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.log_name = f'logs/{self.args.dataset}/{self.start_time}.log'
        self.model_name = f'{self.model_save_path}/{self.start_time}.pth'

        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(f'logs/{self.args.dataset}').mkdir(parents=True, exist_ok=True)

        Logger.init(self.log_name)

        # Get dataloader
        Logger.info('Get dataloader...')
        train_dataloader, valid_dataloader, test_dataloader = self.__get_dataloaders()

        self.train_dataloader: DataLoader = train_dataloader
        self.valid_dataloader: DataLoader = valid_dataloader
        self.test_dataloader: DataLoader = test_dataloader

        self.model = GDN(
            self.num_nodes,
            self.args.slide_window,
            self.args.d_hidden,
            self.args.d_output_hidden,
            self.args.k,
            self.args.num_output_layer,
            dtype=self.args.dtype,
            device=self.args.device
        )
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.loss = MSELoss()

    def __set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)

    def __get_train_and_valid_dataloader(self, train_dataset: TimeDataset, valid_size: float) -> tuple[DataLoader, DataLoader]:
        dataset_size = int(len(train_dataset))
        train_dataset_size = int((1 - valid_size) * dataset_size)
        valid_dataset_size = int(valid_size * dataset_size)

        valid_start_index = random.randrange(train_dataset_size)

        indices = torch.arange(dataset_size)
        train_indices = torch.cat([indices[:valid_start_index], indices[valid_start_index + valid_dataset_size:]])
        valid_indices = indices[valid_start_index:valid_start_index + valid_dataset_size]

        train_subset = Subset(train_dataset, train_indices)
        valid_subset = Subset(train_dataset, valid_indices)

        self.__set_seed()
        train_dataloader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True)

        self.__set_seed()
        valid_dataloader = DataLoader(valid_subset, batch_size=self.args.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader

    def __get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_np = pd.read_csv(f'data/processed/{self.args.dataset}/train.csv').to_numpy()
        test_np = pd.read_csv(f'data/processed/{self.args.dataset}/test.csv').to_numpy()

        train_dataset = TimeDataset(train_np, self.args.slide_window, self.args.slide_stride, 'train', dtype=self.args.dtype)
        test_dataset = TimeDataset(test_np, self.args.slide_window, self.args.slide_stride, 'test', dtype=self.args.dtype)

        self.num_nodes = train_dataset.num_nodes

        train_dataloader, valid_dataloader = self.__get_train_and_valid_dataloader(train_dataset, 0.2)

        self.__set_seed()
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader

    def __train_epoch(self) -> float:
        self.model.train()

        total_train_loss = 0
        for x, y, _ in tqdm(self.train_dataloader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)

            self.optimizer.zero_grad()

            output = self.model(x)

            loss = self.loss(output, y)

            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()

        return total_train_loss / len(self.train_dataloader)

    def __valid_epoch(self, dataloader: DataLoader) -> tuple[float, tuple[Tensor, Tensor, Tensor]]:
        self.model.eval()

        predicted_list = []
        actual_list = []
        label_list = []

        total_valid_loss = 0
        for x, y, label in tqdm(dataloader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            label = label.to(self.args.device)

            with torch.no_grad():
                output = self.model(x)

                loss = self.loss(output, y)

                total_valid_loss += loss.item()

                predicted_list.append(output)
                actual_list.append(y)
                label_list.append(label)

        predicted_tensor = torch.cat(predicted_list, dim=0)
        actual_tensor = torch.cat(actual_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)

        return total_valid_loss / len(self.valid_dataloader), (predicted_tensor, actual_tensor, label_tensor)

    def __train(self):
        best_epoch = -1
        best_train_loss_with_best_epoch = float('inf')
        best_valid_loss = float('inf')
        best_model_weights = copy.deepcopy(self.model.state_dict())
        no_improve_count = 0

        for epoch in tqdm(range(self.args.epoch)):
            train_loss = self.__train_epoch()
            valid_loss, _ = self.__valid_epoch(self.valid_dataloader)

            Logger.info(f'Epoch {epoch + 1}:')
            Logger.info(f' - Train loss: {train_loss:.8f}')
            Logger.info(f' - Valid loss: {valid_loss:.8f}')

            if valid_loss < best_valid_loss:
                best_epoch = epoch + 1

                best_train_loss_with_best_epoch = train_loss
                best_valid_loss = valid_loss

                best_model_weights = copy.deepcopy(self.model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            Logger.info(f' - Current best epoch: {best_epoch}')

            if no_improve_count >= self.args.early_stop:
                break

        torch.save(best_model_weights, self.model_name)

        Logger.info(f'Best epoch: {best_epoch}')
        Logger.info(f' - Train loss: {best_train_loss_with_best_epoch:.8f}')
        Logger.info(f' - Valid loss: {best_valid_loss:.8f}')
        Logger.info(f'Model save to {self.model_name}')

    def __evaluate(self, model_name: str):
        self.model.load_state_dict(torch.load(model_name, weights_only=True))

        _, valid_result = self.__valid_epoch(self.valid_dataloader)
        _, test_result = self.__valid_epoch(self.test_dataloader)

        f1, precision, recall, auc = get_metrics(test_result, valid_result if self.args.report == 'label' else None)

        Logger.info(f'F1 score: {f1:.4f}')
        Logger.info(f'Precision: {precision:.4f}')
        Logger.info(f'Recall: {recall:.4f}')
        Logger.info(f'AUC: {auc:.4f}')

    def run(self):
        if self.args.model_path is None:
            self.__train()
            self.__evaluate(self.model_name)
        else:
            self.__evaluate(f'{self.model_save_path}/{self.args.model_path}')
