import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bsim_datasets.bsmi_dataset import BSIMDataset
from models.param_extractor import ParamExtractorNet
from utils import EarlyStopping, pct_rmse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ivcv = batch['ivcv'].to(device)
        lg = batch['lg'].to(device)
        params = batch['params'].to(device)
        pred = model(ivcv, lg)
        loss = loss_fn(pred, params)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * ivcv.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            ivcv = batch['ivcv'].to(device)
            lg = batch['lg'].to(device)
            params = batch['params'].to(device)
            pred = model(ivcv, lg)
            loss = loss_fn(pred, params)
            total_loss += loss.item() * ivcv.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(params.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return total_loss / len(loader.dataset), preds, trues


def main(args):
    data = np.load(args.data)
    ivcv, lg, params = data['ivcv'], data['lg'], data['params']
    dataset = BSIMDataset(ivcv, lg, params)
    os.makedirs(args.out_dir, exist_ok=True)

    serializable_meta = {k: v.tolist() if isinstance(v, np.ndarray) else v
                         for k, v in dataset.norm_meta.items()}
    with open(os.path.join(args.out_dir, 'norm_meta.json'), 'w') as f:
        json.dump(serializable_meta, f, indent=2)

    n = len(dataset)
    n_val = int(0.1*n)
    n_train = n - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = ParamExtractorNet(hidden_size=args.hidden_size, num_hidden=args.num_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    early = EarlyStopping(patience=args.patience)

    best_val = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_preds, val_trues = eval_model(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        val_pct = pct_rmse(val_preds, val_trues)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_pct_rmse={val_pct:.4f} time={(time.time()-t0):.1f}s")

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            print('Saved best model.')

        early.step(val_loss)
        if early.early_stop:
            print("Early stopping triggered.")
            break

    print("Training finished. Best val loss:", best_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/demo_data.npz')
    parser.add_argument('--out_dir', type=str, default='experiments/exp_001')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--num_hidden', type=int, default=3)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    main(args)

