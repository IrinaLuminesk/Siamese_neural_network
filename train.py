import argparse
import os
import random
import numpy as np
from tqdm import tqdm
# from Aug.BatchWiseAug import BatchWiseAug
# from Metrics.MetricCal import MetricCal
# from learning_rate_helper.learning_rate import PiecewiseScheduler
# from model_builder.model import SiameseModel
from dataset_helper.DatasetLoader import DatasetLoader
# from utils.MetricCalV2 import MetricCalV2
# from utils.Utilities import Get_Min_EER, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric2, YAML_Reader, get_mean_std

import torch
import torch.nn as nn
import torch.optim as optim

# from timm.loss.cross_entropy import SoftTargetCrossEntropy

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

# def train(epoch: int, end_epoch: int, model, loader, criterion, optimizer, device):
#     model.train()
#     # freeze_bn(model)
#     metrics = MetricCalV2(device=device)
#     for inputs1, inputs2, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
#                                 format(epoch, end_epoch)):

#         inputs1, inputs2 = inputs1.to(device, non_blocking=True), inputs2.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)
        
#         optimizer.zero_grad()
#         outputs1, outputs2 = model(inputs1, inputs2)
#         loss = criterion(outputs1, outputs2, targets)
#         loss.backward()
#         optimizer.step()

#         metrics.update_train(loss=loss, outputs1=outputs1, outputs2=outputs2, targets=targets)
#     metrics.compute_fpr_tpr_thresholds()
#     return metrics

# def validate(epoch, end_epoch, model, loader, criterion, device):
#     model.eval()
#     metrics = MetricCalV2(device=device)
#     with torch.no_grad():
#         for inputs1, inputs2, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
#                                 format(epoch, end_epoch)):
#             inputs1, inputs2 = inputs1.to(device, non_blocking=True), inputs2.to(device, non_blocking=True)
#             targets = targets.to(device, non_blocking=True)
            
#             outputs1, outputs2 = model(inputs1, inputs2)
#             loss = criterion(outputs1, outputs2, targets)
#             metrics.update_test(loss=loss, outputs1=outputs1, outputs2=outputs2, targets=targets)
#     metrics.compute_fpr_tpr_thresholds()
#     return metrics

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
    Learning_rate_para = config["TRAIN"]["LEARNING_RATE"]["PieceWise"]

    #Optional
    save_checkpoint = config["TRAIN"]["OPTIONAL"]["SAVE_CHECKPOINT"]
    save_best = config["TRAIN"]["OPTIONAL"]["SAVE_BEST"]
    save_metrics = config["TRAIN"]["OPTIONAL"]["SAVE_METRICS"]
    checkpoint_path = config["TRAIN"]["OPTIONAL"]["CHECKPOINT_PATH"]
    best_path = config["TRAIN"]["OPTIONAL"]["BEST_PATH"]
    metrics_path = config["TRAIN"]["OPTIONAL"]["METRICS_PATH"]
    
    set_seed()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # train_data = DatasetLoader(path=train_path, batch_size=batch_size, number_of_sample=train_sample,std=std, mean=mean, img_size=img_size, transform=enabled_transform)
    # test_data = DatasetLoader(path=test_path, batch_size=batch_size, number_of_sample=test_sample, std=std, mean=mean, img_size=img_size)

    # training_loader = train_data.dataset_loader("train")
    # testing_loader = test_data.dataset_loader("test")

    # model = SiameseModel(model_type=model_type, embedding_dim=embedding_dim).to(device)
    # # freeze_bn(model)
    # eval_criterion = nn.CosineEmbeddingLoss(margin=0.5)
    # train_criterion = nn.CosineEmbeddingLoss(margin=0.5)
    # optimizer = optim.AdamW(model.parameters(), lr=Learning_rate_para["MAX_LR"], weight_decay=1e-2)


    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=end_epoch,
    #     eta_min=1e-6
    # )
    # print("Training using CosineAnnealingLR")
    # best_eer = 999999999999

    # if resume == True:
    #     begin_epoch = Loading_Checkpoint(path=checkpoint_path,
    #                                      model=model,
    #                                      optimizer=optimizer,
    #                                      scheduler=scheduler,
    #                                      device=device)
    #     best_eer = Get_Min_EER(metrics_path)
    # UNFREEZE_EPOCH = 20
    # for epoch in range(begin_epoch, end_epoch):
    #     # if epoch == UNFREEZE_EPOCH - 1:
    #     #     freeze_unfreeze_backbone(model, True)
    #     #     optimizer, scheduler = recreate_optimizer_and_scheduler(model=model, num_epochs=end_epoch, epoch=epoch)
    #     train_metrics = train(epoch, 
    #                             end_epoch,
    #                             model=model, 
    #                             loader=training_loader, 
    #                             criterion=train_criterion, 
    #                             optimizer=optimizer, 
    #                             device=device)
    #     train_loss, train_ROC_AUC = train_metrics.avg_cosemb_loss, train_metrics.ROC_AUC
    #     train_EER = train_metrics.EER
    #     train_TAR_and_FAR_1p = train_metrics.tar_at_far(0.01)
    #     train_TAR_and_FAR_01p = train_metrics.tar_at_far(0.001)
    #     scheduler.step()
    #     print()
    #     if epoch % 4 == 0: #Chỉ tạo mới sau 5 epoch, chia hết cho 4 vì epoch bắt đầu là 0 
    #         train_data.regenerate_pair() #Tạo mới pairs
        
    #     val_metrics = validate(epoch, end_epoch, model, testing_loader, eval_criterion, device)
    #     val_loss, val_ROC_AUC = val_metrics.avg_cosemb_loss, val_metrics.ROC_AUC
    #     val_EER = val_metrics.EER
    #     val_TAR_and_FAR_1p = val_metrics.tar_at_far(0.01)
    #     val_TAR_and_FAR_01p = val_metrics.tar_at_far(0.001)
    #     print()

    #     if save_checkpoint == True:
    #         Saving_Checkpoint(epoch=epoch, 
    #                         model=model, 
    #                         optimizer=optimizer, 
    #                         scheduler=scheduler,
    #                         last_epoch=epoch, 
    #                         path=checkpoint_path)

    #     print("Epoch [{0}/{1}]: Training loss: {2}, ROC AUC: {3}, EER: {4}, TAR @ FAR=1%: {5}".
    #         format(epoch, end_epoch, train_loss, train_ROC_AUC, train_EER, train_TAR_and_FAR_1p))
    #     print("Epoch [{0}/{1}]: Validation loss: {2}, ROC AUC: {3}, EER: {4}, TAR @ FAR=1%: {5}".
    #         format(epoch, end_epoch, val_loss, val_ROC_AUC, val_EER, val_TAR_and_FAR_1p))
    #     if val_EER < best_eer:
    #         if save_best == True:
    #             print("Validation EER improve from {0} to {1} at epoch {2}. Saving best result".
    #                 format(round(best_eer, 4), round(val_EER, 4),  epoch))
    #             Saving_Best(model, best_path)
    #         else:
    #             print("Validation accuracy improve from {0} to {1} at epoch {2}".
    #                 format(round(best_eer, 4), round(val_EER, 4),  epoch))
    #         best_eer = val_EER
    #         epochs_no_improve = 0  # reset patience
    #     else:
    #         epochs_no_improve += 1
    #     if save_metrics:
    #         Saving_Metric2(epoch=epoch, 
    #                        train_loss=train_loss,
    #                        train_ROC_AUC=train_ROC_AUC,
    #                        train_EER=train_EER,
    #                        train_TAR_and_FAR_1p=train_TAR_and_FAR_1p,
    #                        train_TAR_and_FAR_01p=train_TAR_and_FAR_01p, 
    #                        val_loss=val_loss,
    #                        val_ROC_AUC=val_ROC_AUC,
    #                        val_EER=val_EER,
    #                        val_TAR_and_FAR_1p=val_TAR_and_FAR_1p,
    #                        val_TAR_and_FAR_01p=val_TAR_and_FAR_01p, 
    #                        path=metrics_path)
    #     if epochs_no_improve >= patience and early_stopping == True:
    #         print("Early stopping triggered at epoch {0}".format(epoch))
    #         break
    #     print()

    
if __name__ == '__main__':
    main()