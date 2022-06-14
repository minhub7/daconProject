"""Recorder
"""

import os
import sys

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import logging
import torch
import csv


class Recorder():

    def __init__(self,
                 record_dir: str,
                 model: object,
                 optimizer: object,
                 scheduler: object,
                 logger: logging.RootLogger=None):
        """Recorder 초기화
            
        Args:

        Note:
        """
        
        self.record_dir = record_dir
        self.plot_dir = os.path.join(record_dir, 'plots')
        self.record_filepath = os.path.join(self.record_dir, 'record.csv')
        self.weight_path = os.path.join(record_dir, 'model.pt')

        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        os.makedirs(self.plot_dir, exist_ok=True)

    def set_model(self, model: 'model'):
        self.model = model

    def set_logger(self, logger: logging.RootLogger):
        self.logger = logger

    def create_record_directory(self):
        """
        record 경로 생성
        """
        os.makedirs(self.record_dir, exist_ok=True)

        msg = f"Create directory {self.record_dir}"
        self.logger.info(msg) if self.logger else None

    def add_row(self, row_dict: dict):
        """Epoch 단위 성능 적재

        Args:
            row (list): 

        """

        fieldnames = list(row_dict.keys())

        with open(self.record_filepath, newline='', mode='a') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(row_dict)
            msg = f"Write row {row_dict['epoch_index']}"
            self.logger.info(msg) if self.logger else None

    def save_weight(self, epoch: int)-> None:
        """Weight 저장
            amp 추가
        Args:
            loss (float): validation loss
            model (`model`): model
        
        """
        check_point = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),  # 'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(check_point, self.weight_path)
        # torch.save(self.model.state_dict(), self.weight_path)
        msg = f"Recorder, epoch {epoch} Model saved: {self.weight_path}"
        self.logger.info(msg) if self.logger else None

    
    def save_plot(self, plots: list):

        record_df = pd.read_csv(self.record_filepath)
        current_epoch = record_df['epoch_index'].max()
        epoch_range = list(range(0, current_epoch+1))
        color_list = ['red', 'blue']  # train, val

        for plot_name in plots:
            columns = [f'train_{plot_name}', f'val_{plot_name}']

            fig = plt.figure(figsize=(20, 8))
            
            for id_, column in enumerate(columns):
                values = record_df[column].tolist()
                plt.plot(epoch_range, values, marker='.', c=color_list[id_], label=column)
             
            plt.title(plot_name, fontsize=15)
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel(plot_name)
            plt.xticks(epoch_range, [str(i) for i in epoch_range])
            plt.close(fig)
            fig.savefig(os.path.join(self.plot_dir, plot_name +'.png'))