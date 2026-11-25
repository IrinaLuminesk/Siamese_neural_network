import torch
from tqdm import tqdm
import yaml
import numpy as np
import os
import pandas as pd

def Create_Folder(path):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

def YAML_Reader(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def Saving_Checkpoint(epoch, model, optimizer, scheduler, last_epoch, path):
    Create_Folder(path=path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        "last_epoch": last_epoch
    }, path)

def Saving_Best(model, path):
    Create_Folder(path=path)
    torch.save(model.state_dict(), path)

def Saving_Metric(epoch, train_acc, train_loss, top1_val_acc, top5_val_acc, val_loss, path):
    Create_Folder(path=path)
    if os.path.exists(path):
        metrics_df = pd.read_csv(path)
    else:
        metrics_df = pd.DataFrame({
            'epoch': pd.Series(dtype='int'),
            'train_loss': pd.Series(dtype='float'),
            'train_acc': pd.Series(dtype='float'),
            'val_loss': pd.Series(dtype='float'),
            'top1_val_acc': pd.Series(dtype='float'),
            'top5_val_acc': pd.Series(dtype='float'),
            'lr': pd.Series(dtype='float')
        })
    new_row = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'top1_val_acc': top1_val_acc,
        'top5_val_acc': top5_val_acc,
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
    metrics_df.to_csv(path, index=False)

def Saving_Metric2(epoch, 
                   train_loss,
                   train_EER,
                   train_EER_threshold,
                   train_ROC_AUC,
                   train_acc,
                   train_prec,
                   train_rec,
                   train_f1,
                   val_loss,
                   val_EER,
                   val_EER_threshold,
                   val_ROC_AUC,
                   val_acc,
                   val_prec,
                   val_rec,
                   val_f1,
                   path):
    Create_Folder(path=path)
    if os.path.exists(path):
        metrics_df = pd.read_csv(path)
    else:
        metrics_df = pd.DataFrame({
            'epoch': pd.Series(dtype='int'),
            'train_loss': pd.Series(dtype='float'),
            'train_EER': pd.Series(dtype='float'),
            'train_EER_threshold': pd.Series(dtype='float'),
            'train_ROC_AUC': pd.Series(dtype='float'),
            'train_acc': pd.Series(dtype='float'),
            'train_prec': pd.Series(dtype='float'),
            'train_rec': pd.Series(dtype='float'),
            'train_f1': pd.Series(dtype='float'),
            'val_loss': pd.Series(dtype='float'),
            'val_EER': pd.Series(dtype='float'),
            'val_EER_threshold': pd.Series(dtype='float'),
            'val_ROC_AUC': pd.Series(dtype='float'),
            'val_acc': pd.Series(dtype='float'),
            'val_prec': pd.Series(dtype='float'),
            'val_rec': pd.Series(dtype='float'),
            'val_f1': pd.Series(dtype='float')
        })
    new_row = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_EER': train_EER,
        'train_EER_threshold': train_EER_threshold,
        'train_ROC_AUC': train_ROC_AUC,
        'train_acc': train_acc,
        'train_prec': train_prec,
        'train_rec': train_rec,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_EER': val_EER,
        'val_EER_threshold': val_EER_threshold,
        'val_ROC_AUC': val_ROC_AUC,
        'val_acc': val_acc,
        'val_prec': val_prec,
        'val_rec': val_rec,
        'val_f1': val_f1
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
    metrics_df.to_csv(path, index=False)

def Loading_Checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('last_epoch', 0) + 1
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch

def Get_Min_EER(path):
    df = pd.read_csv(path)

    best_eer = df['val_EER'].min() #Ngược với classification

    return best_eer 

def get_num_workers(transform_heavy=True):
    # 1. Detect CPU cores
    cpu_cores = os.cpu_count() or 1

    # 2. Detect GPU count
    gpu_count = torch.cuda.device_count()

    # 3. Select scaling factor
    # Heavy transforms: ElasticTransform, RandomAffine, Perspective, etc.
    if transform_heavy:
        scale = 1.5
    else:
        scale = 1.0

    # 4. Handle GPU or CPU-only case
    if gpu_count > 0:
        # Workers per GPU
        workers = int((cpu_cores / gpu_count) * scale)
    else:
        # CPU-only system, use more workers since CPU handles everything
        workers = int(cpu_cores * 0.75)

    # 5. Clamp values
    workers = max(2, workers)        # never less than 2
    workers = min(workers, cpu_cores - 1)  # never use ALL CPU cores

    return workers