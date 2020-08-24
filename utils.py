import torch
import torch.nn.functional as F
import numpy as np


def evaluate(model, dataloader, CUDA=True):
    model.eval()
    if CUDA:
        model.cuda()

    predictions = list()
    targets_list = list()
    losses = list()
    for img, targets in dataloader:
        targets_list.append(targets["T"].item())
        if CUDA:
            img = img.cuda()
            targets["T"] = targets["T"].cuda()
        with torch.no_grad():
            output = model(img)
            predictions.append(output.cpu().item())
            loss = F.mse_loss(output.flatten(), targets["T"])
            losses.append(loss.item())
    return losses, targets_list, predictions


def format_mse(targets, losses, nbins=20):
    losses = [x for _,x in sorted(zip(targets, losses))]
    targets = sorted(targets)
    new_tar, new_loss = ([], [])
    binsize = len(targets) // nbins
    width = binsize * np.diff(targets).mean()
    for i in range(nbins):
        new_tar.append(np.median(targets[i * binsize: (i+1) * binsize]))
        new_loss.append(np.mean(losses[i * binsize: (i+1) * binsize]))

    return new_tar, new_loss, width

