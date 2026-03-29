"""Trainer for DeepPhys."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class DeepPhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        # Fix 1: ENABLE aus dem TDCM-Block lesen
        self.use_tdcm = config.MODEL.TDCM.get("ENABLE", False)
        tdcm_config = {k: v for k, v in config.MODEL.TDCM.items() if k != "ENABLE"}

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = DeepPhys(
                img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H,
                use_tdcm=self.use_tdcm,
                tdcm_config=tdcm_config
            ).to(self.device)

            # Fix 2: DataParallel nur wenn GPUs vorhanden
            if config.NUM_OF_GPU_TRAIN > 0:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)

        elif config.TOOLBOX_MODE == "only_test":
            self.model = DeepPhys(
                img_size=config.TEST.DATA.PREPROCESS.RESIZE.H,
                use_tdcm=self.use_tdcm,
                tdcm_config=tdcm_config
            ).to(self.device)

            if config.NUM_OF_GPU_TRAIN > 0:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("DeepPhys trainer initialized in incorrect toolbox mode!")

    def _prepare_batch(self, data, labels):
        """Bereitet Input-Tensoren je nach TDCM-Modus vor.

        Standard-Modus (use_tdcm=False):
            data:   [N, D, C, H, W] -> [N*D, C, H, W]
            labels: [N, D]          -> [N*D, 1]

        TDCM-Modus (use_tdcm=True):
            data:   [N, D, C, H, W] -> unveraendert
            labels: [N, D]          -> [N, D, 1]
        """
        N, D, C, H, W = data.shape
        if self.use_tdcm:
            data_out   = data                    # [N, D, C, H, W]
            labels_out = labels.view(N, D, 1)    # [N, D, 1]
        else:
            data_out   = data.view(N * D, C, H, W)  # [N*D, C, H, W]
            labels_out = labels.view(-1, 1)           # [N*D, 1]
        return data_out, labels_out

    def _flatten_outputs(self, pred, labels, N, D):
        """Im TDCM-Modus: [N, D, 1] -> [N*D, 1] fuer einheitliches Speichern."""
        if self.use_tdcm:
            pred   = pred.view(N * D, 1)
            labels = labels.view(N * D, 1)
        return pred, labels

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                data, labels = self._prepare_batch(data, labels)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)

                assert pred_ppg.shape == labels.shape, \
                    f"Shape mismatch: pred={pred_ppg.shape}, labels={labels.shape}"

                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                lrs.append(self.scheduler.get_last_lr())
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

                train_loss.append(loss.item())
                tbar.set_postfix({
                    "loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]["lr"]
                })

            mean_training_losses.append(np.mean(train_loss))
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))

        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                data_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                data_valid, labels_valid = self._prepare_batch(data_valid, labels_valid)

                pred_ppg_valid = self.model(data_valid)

                assert pred_ppg_valid.shape == labels_valid.shape, \
                    f"Shape mismatch: pred={pred_ppg_valid.shape}, labels={labels_valid.shape}"

                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item())

        return np.mean(np.asarray(valid_loss))

    def test(self, data_loader):
        """Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")

        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")

        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), \
                    test_batch[1].to(self.config.DEVICE)

                N, D, C, H, W = data_test.shape
                data_test, labels_test = self._prepare_batch(data_test, labels_test)

                pred_ppg_test = self.model(data_test)

                # Im TDCM-Modus: [N, D, 1] -> [N*D, 1] fuer chunk-weises Speichern
                pred_ppg_test, labels_test = self._flatten_outputs(pred_ppg_test, labels_test, N, D)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test    = labels_test.cpu()
                    pred_ppg_test  = pred_ppg_test.cpu()

                for idx in range(N):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index]      = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        """Saves the model to disk."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir,
            self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)