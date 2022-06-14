"""Trainer
"""

from modules.metrics import ConfMatrix
from modules.losses import *
from modules.utils import *
from modules.datasets import batch_transform
from modules.utils import generate_unsup_data, label_onehot
import torch
from time import time
import pandas as pd
from tqdm import tqdm

class Trainer():

    def __init__(self, model, ema, data_loader, optimizer, device, config, logger, interval=100):
        self.model = model
        self.ema = ema
        self.data_loader = data_loader
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.interval = interval
        self.mIoU = 0

        # History
        self.loss_sum = 0  # Epoch loss sum
        self.loss_mean = 0  # Epoch loss mean
        self.score_dict = dict()  # metric score
        self.elapsed_time = 0

    def train(self, train_l_loader, train_u_loader):
        train_l_dataset = iter(train_l_loader)  # BuildDataLoader로 불려진 train_l_loader를 iter() 반복자 객체로 선언
        train_u_dataset = iter(train_u_loader)
        self.model.train()  # model train 모드로 설정
        self.ema.model.train()  # model train 모드로 설정

        train_size = len(train_l_loader)   # dataset의 크기만큼 for loop 수행
        start_timestamp = time()
        l_conf_mat = ConfMatrix(self.data_loader.num_segments)  # data_loader.num_segments = 5

        for i in tqdm(range(train_size), desc="Training", ncols=100):
            # iter 객체의 next 메소드를 통해 __getitem__ 메소드 호출
            # print("\nStart Train")
            train_l_data, train_l_label = train_l_dataset.next()
            train_l_data, train_l_label = train_l_data.to(self.device), train_l_label.to(self.device)
            train_u_data = train_u_dataset.next()
            train_u_data = train_u_data.to(self.device)
            # print(f"\ntrain_u_loader file name is {train_u_loader.dataset.filename}")
            self.optimizer.zero_grad()

            # print("Start generate pseudo-labels")
            # generate pseudo-labels
            with torch.no_grad():  # no_grad()를 사용하는 이유는 unlabeled_data 라서
                pred_u, _ = self.ema.model(train_u_data)
                # print(f"\nprint train_u_data shape: {train_u_data.shape}")
                #
                # print("Start augmentation")

                pred_u_large_raw = F.interpolate(pred_u, size=train_u_data.shape[2:], mode='bilinear', align_corners=True)
                # print(f"\nprint pred_u_large_raw shape: {pred_u_large_raw.shape}")
                # softmax 함수로 각 행의 원소를 확률값으로 만든 후 max값 추출 - 어떤 labels에 해당하는 지 판단 가능
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)  # output (max, max_indices)
                # print(f"pseudo_logits, pseudo_labels is {pseudo_logits} {pseudo_labels}")

                # for logit, label in zip(pseudo_logits, pseudo_labels):
                #     print(f"logit = {logit[0]}\n logit_size = {logit[0].shape} ,\n label = {label[0]}\n label_size = {label[0].shape}")
                #     break

                # print("Start Batch transform")
                # random scale images first
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                    self.data_loader.crop_size, self.data_loader.scale_size, apply_augmentation=False)

                # apply mixing strategy: cutout, cutmix or classmix
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=self.config['apply_aug'])

                # apply augmentation: color jitter + flip + gaussian blur
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                    self.data_loader.crop_size, (1.0, 1.0), apply_augmentation=True)

            # print("Start loss calculate")

            # generate labelled and unlabelled data loss
            pred_l, rep_l = self.model(train_l_data)
            pred_l_large = F.interpolate(pred_l, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

            pred_u, rep_u = self.model(train_u_aug_data)
            pred_u_large = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

            rep_all = torch.cat((rep_l, rep_u))  # rep_l과 rep_u를 합침
            pred_all = torch.cat((pred_l, pred_u))

            # supervised-learning loss
            sup_loss = compute_supervised_loss(pred_l_large, train_l_label)

            # unsupervised-learning loss
            unsup_loss = compute_unsupervised_loss(pred_u_large, train_u_aug_label, train_u_aug_logits,
                                                   self.config['strong_threshold'])

            # apply regional contrastive loss
            if self.config['apply_reco']:
                with torch.no_grad():
                    train_u_aug_mask = train_u_aug_logits.ge(self.config['weak_threshold']).float()
                    mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                    mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                    label_l = F.interpolate(label_onehot(train_l_label, self.data_loader.num_segments),
                                            size=pred_all.shape[2:], mode='nearest')
                    label_u = F.interpolate(label_onehot(train_u_aug_label, self.data_loader.num_segments),
                                            size=pred_all.shape[2:], mode='nearest')
                    label_all = torch.cat((label_l, label_u))

                    prob_l = torch.softmax(pred_l, dim=1)
                    prob_u = torch.softmax(pred_u, dim=1)
                    prob_all = torch.cat((prob_l, prob_u))

                reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, self.config['strong_threshold'],
                                              self.config['temp'], self.config['num_queries'], self.config['num_negatives'])
            else:
                reco_loss = torch.tensor(0.0)

            loss = sup_loss + unsup_loss + reco_loss
            loss.backward()
            self.optimizer.step()
            self.ema.update(self.model)
            l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())

            # History
            self.loss_sum += loss.item()

            # Logging
            if i % self.interval == 0:
                msg = f"batch: {i}/{train_size} loss: {loss.item()}"
                self.logger.info(msg)
                
        # Epoch history
        self.loss_mean = self.loss_sum / train_size  # Epoch loss mean
        self.mIoU, _ = l_conf_mat.get_metrics()
        self.score_dict['mIoU'] = self.mIoU

        # Elapsed time
        end_timestamp = time()
        self.elapsed_time = end_timestamp - start_timestamp

    def valid(self, valid_l_loader):
        valid_epoch = len(valid_l_loader)
        with torch.no_grad():
            self.ema.model.eval()
            valid_dataset = iter(valid_l_loader)
            valid_conf_mat = ConfMatrix(self.data_loader.num_segments)
            for i in tqdm(range(valid_epoch), desc="Validation", ncols=100):
                valid_data, valid_label = valid_dataset.next()
                valid_data, valid_label = valid_data.to(self.device), valid_label.to(self.device)

                pred, _ = self.ema.model(valid_data)
                pred_u_large_raw = F.interpolate(pred, size=valid_label.shape[1:], mode='bilinear', align_corners=True)
                valid_conf_mat.update(pred_u_large_raw.argmax(1).flatten(), valid_label.flatten())
        self.mIoU, _ = valid_conf_mat.get_metrics()
        self.score_dict['mIoU'] = self.mIoU

    def inference(self, test_loader, save_path, sample_submission):
        # batch size of the test loader should be 1
        class_map = {0: 'container_truck', 1: 'forklift', 2: 'reach_stacker', 3: 'ship'}
        test_size = len(test_loader)
        file_names = []
        classes = []
        predictions = []
        with torch.no_grad():
            self.ema.model.eval()
            test_dataset = iter(test_loader)
            for i in tqdm(range(test_size), desc="Inference", ncols=100):
                test_data, img_size, filename = test_dataset.next()
                test_data = test_data.to(self.device)
                pred, _ = self.ema.model(test_data)
                pred_u_large_raw = F.interpolate(pred, size=img_size[0].tolist(), mode='bilinear', align_corners=True)
                class_num = pred_u_large_raw[0].sum(dim=(1, 2))[1:].argmax().item()
                class_of_image = class_map[class_num]
                # mask를 계산하는 부분
                class_mask = (pred_u_large_raw[0][class_num + 1] - pred_u_large_raw[0][0] > 0).int().cpu().numpy()
                coverted_coordinate = mask_to_coordinates(class_mask)
                file_names.append(filename[0])
                classes.append(class_of_image)
                predictions.append(coverted_coordinate)

        submission_df = pd.DataFrame({'file_name': file_names, 'class': classes, 'prediction': predictions})
        submission_df = pd.merge(sample_submission['file_name'], submission_df, left_on='file_name', right_on='file_name', how='left')
        submission_df.to_csv(save_path, index=False, encoding='utf-8')

    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.mIoU = 0
        self.score_dict = dict()
        self.elapsed_time = 0