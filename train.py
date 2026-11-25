import argparse
from cProfile import label
import os
import random
import numpy as np
from tqdm import tqdm
from model_builder.baseline_model import Model
from dataset_helper.DatasetLoader import DatasetLoader
from loss_helper.LossFunction import LossFunction
from utils.MetricCal import MetricCal
from utils.Utilities import Get_Min_EER, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric2, YAML_Reader

import torch
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argparse example")
    
    # Add arguments
    parser.add_argument(
    "--cfg",
    type=str,
    default="config/default_config.yaml",
    help="Config file used to train the model (default: config/default_config.yaml)"
    )
    args = parser.parse_args()
    config = YAML_Reader(args.cfg)
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(epoch: int, end_epoch: int, model, pair_type, loader, criterion, optimizer, device):
    model.train()
    metrics = MetricCal(device=device, pair_type=pair_type)
    for batch in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
        optimizer.zero_grad()
        match pair_type:
            case 1:
                imgs1, imgs2, labels = batch
                imgs1, imgs2 = imgs1.to(device, non_blocking=True), imgs2.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                embs1, embs2 = model(x1=imgs1, x2=imgs2)
                loss = criterion(embs1=embs1, embs2=embs2, labels=labels)
                metrics.update(loss=loss, embs1=embs1, embs2=embs2, labels=labels)
            case 2:
                anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_imgs = anchor_imgs.to(device, non_blocking=True)
                positive_imgs, negative_imgs = positive_imgs.to(device, non_blocking=True), negative_imgs.to(device, non_blocking=True)
                embs1, embs2, embs3 = model(x1=anchor_imgs, x2=positive_imgs, x3=negative_imgs)
                loss = criterion(embs1=embs1, embs2=embs2, embs3=embs3)
            case _:
                raise ValueError(
                    f"Invalid pair_type: {pair_type}. Expected 1 or 2."
                )
        loss.backward()
        optimizer.step()
    return metrics

def validate(epoch, end_epoch, model, pair_type, loader, criterion, device):
    model.eval()
    metrics = MetricCal(device=device, pair_type=pair_type)
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            match pair_type:
                case 1:
                    imgs1, imgs2, labels = batch
                    imgs1, imgs2 = imgs1.to(device, non_blocking=True), imgs2.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    embs1, embs2 = model(x1=imgs1, x2=imgs2)
                    loss = criterion(embs1=embs1, embs2=embs2, labels=labels)
                    metrics.update(loss=loss, embs1=embs1, embs2=embs2, labels=labels)
                case 2:
                    anchor_imgs, positive_imgs, negative_imgs = batch
                    anchor_imgs = anchor_imgs.to(device, non_blocking=True)
                    positive_imgs, negative_imgs = positive_imgs.to(device, non_blocking=True), negative_imgs.to(device, non_blocking=True)
                    embs1, embs2, embs3 = model(x1=anchor_imgs, x2=positive_imgs, x3=negative_imgs)
                    loss = criterion(embs1=embs1, embs2=embs2, embs3=embs3)
                case _:
                    raise ValueError(
                    f"Invalid pair_type: {pair_type}. Expected 1 or 2.")
    return metrics

def main():
    config = parse_args()


    #Data parameters
    # root_path = config["DATASET"]["ROOT_FOLDER"]
    train_path = config["DATASET"]["TRAIN_FOLDER"]
    train_sample = int(config["DATASET"]["TRAIN_SAMPLE"])
    val_path = config["DATASET"]["VAL_FOLDER"]
    val_sample = int(config["DATASET"]["VAL_SAMPLE"])
    mean = config["TRAIN"]["DATA"]["MEAN"]
    std = config["TRAIN"]["DATA"]["STD"]
    batch_size = config["TRAIN"]["DATA"]["BATCH_SIZE"]
    

    #Training parameters
    img_size = config["TRAIN"]["DATA"]["IMAGE_SIZE"]
    enabled_transform = config["TRAIN"]["TRANSFORM"]
    begin_epoch = config["TRAIN"]["TRAIN_PARA"]["BEGIN_EPOCH"] 
    end_epoch = config["TRAIN"]["TRAIN_PARA"]["END_EPOCH"]
    resume = config["TRAIN"]["TRAIN_PARA"]["RESUME"]
    early_stopping = int(config["TRAIN"]["TRAIN_PARA"]["EARLY_STOPPING"])
    patience = config["TRAIN"]["TRAIN_PARA"]["PATIENCE"]
    epochs_no_improve = 0
    model_type = int(config["TRAIN"]["TRAIN_PARA"]["MODEL_TYPE"])
    embedding_dim = int(config["TRAIN"]["TRAIN_PARA"]["EMBEDDING_DIM"])

    #Learning_rate
    # Learning_rate_para = config["TRAIN"]["LEARNING_RATE"]["PieceWise"]

    #Optional
    save_checkpoint = config["TRAIN"]["OPTIONAL"]["SAVE_CHECKPOINT"]
    save_best = config["TRAIN"]["OPTIONAL"]["SAVE_BEST"]
    save_metrics = config["TRAIN"]["OPTIONAL"]["SAVE_METRICS"]
    checkpoint_path = config["TRAIN"]["OPTIONAL"]["CHECKPOINT_PATH"]
    best_path = config["TRAIN"]["OPTIONAL"]["BEST_PATH"]
    metrics_path = config["TRAIN"]["OPTIONAL"]["METRICS_PATH"]
    
    set_seed()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = DatasetLoader(
        path=train_path,
        pair_type=1,
        batch_size=batch_size,
        number_of_sample=train_sample,
        std=std,
        mean=mean,
        img_size=img_size,
        enable_transform=enabled_transform)
    training_loader = train_data.dataset_loader("train")
    
    val_data = DatasetLoader(
        path=val_path,
        pair_type=1,
        batch_size=batch_size,
        number_of_sample=val_sample,
        std=std,
        mean=mean,
        img_size=img_size,
        enable_transform=False)
    val_loader = val_data.dataset_loader("val")

    model = Model(model_type=1, pair_type=1, embedding_dim=embedding_dim).to(device)

    val_criterion = LossFunction(pair_type=1, margin=1)
    train_criterion = LossFunction(pair_type=1, margin=1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=end_epoch,
        eta_min=1e-6
    )
    print("Training using CosineAnnealingLR")
    best_eer = 999999999999

    if resume == True:
        begin_epoch = Loading_Checkpoint(path=checkpoint_path,
                                         model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         device=device)
        best_eer = Get_Min_EER(metrics_path)
    for epoch in range(begin_epoch, end_epoch):
        train_metrics = train(epoch, 
                                end_epoch,
                                model=model,
                                pair_type=1, 
                                loader=training_loader, 
                                criterion=train_criterion, 
                                optimizer=optimizer, 
                                device=device)
        train_data.regenerate_pairs()
        train_EER, train_EER_threshold = train_metrics.get_threshold()
        train_loss = train_metrics.avg_loss
        scheduler.step()
        print()
        
        val_metrics = validate(epoch, 
                               end_epoch, 
                               model=model, 
                               pair_type=1, 
                               loader=val_loader, 
                               criterion=val_criterion, 
                               device=device)
        val_EER, val_EER_threshold = val_metrics.get_threshold()
        val_loss = val_metrics.avg_loss
        print()

        if save_checkpoint == True:
            Saving_Checkpoint(epoch=epoch, 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler,
                            last_epoch=epoch, 
                            path=checkpoint_path)

        print("Epoch [{0}/{1}]: Training loss: {2}, EER: {3}".
            format(epoch, end_epoch, train_loss, train_EER))
        print("Epoch [{0}/{1}]: Validation loss: {2}, EER: {3}".
            format(epoch, end_epoch, val_loss, val_EER))
        if val_EER < best_eer:
            if save_best == True:
                print("Validation EER improve from {0} to {1} at epoch {2}. Saving best result".
                    format(round(best_eer, 4), round(val_EER, 4),  epoch))
                Saving_Best(model, best_path)
            else:
                print("Validation accuracy improve from {0} to {1} at epoch {2}".
                    format(round(best_eer, 4), round(val_EER, 4),  epoch))
            best_eer = val_EER
            epochs_no_improve = 0  # reset patience
        else:
            epochs_no_improve += 1
        if save_metrics:
            Saving_Metric2(epoch=epoch, 
                           train_loss=train_loss,
                           train_EER=train_EER,
                           train_EER_threshold=train_EER_threshold,
                           train_ROC_AUC=train_metrics.ROC_AUC,
                           train_acc=train_metrics.Accuracy_score,
                           train_prec=train_metrics.Precision_score,
                           train_rec=train_metrics.Recall_score,
                           train_f1=train_metrics.F1_score,
                           val_loss=val_loss,
                           val_EER=val_EER,
                           val_EER_threshold=val_EER_threshold,
                           val_ROC_AUC=val_metrics.ROC_AUC,
                           val_acc=val_metrics.Accuracy_score,
                           val_prec=val_metrics.Precision_score,
                           val_rec=val_metrics.Recall_score,
                           val_f1=val_metrics.F1_score,
                           path=metrics_path
                        )
        if epochs_no_improve >= patience and early_stopping == True:
            print("Early stopping triggered at epoch {0}".format(epoch))
            break
        print()

    
if __name__ == '__main__':
    main()