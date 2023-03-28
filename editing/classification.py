import numpy as np
import torch
from torch import nn, optim
from tqdm.auto import tqdm

import utilities.data as dutils
import utilities.utilities as utils


def get_class_indices(classes, target_classes):
    target_class_indices = [classes.index(target_class) for target_class in target_classes]
    return target_class_indices


def get_class_counts(loader, classes, target_classes=None, ratio=False):
    if target_classes is None:
        target_classes = classes
        target_class_indices = list(range(len(classes)))
    else:
        target_class_indices = get_class_indices(classes, target_classes)
    total = 0
    counts = torch.zeros(len(target_class_indices), dtype=torch.int32)
    for _, targets in tqdm(iter(loader)):
        total += targets.shape[0]
        counts += targets[:, target_class_indices].sum(dim=0)
    if ratio:
        counts = [count.item() / total for count in counts]
    else:
        counts = [(count.item() / total) for count in counts]
    counts = dict(zip(target_classes, counts))
    return counts


def balance_output(loss, targets, target_classes, class_weights):
    loss_balanced = torch.zeros_like(loss).to(loss)
    for j, target_class in enumerate(target_classes):
        weight = class_weights[target_class]
        loss_positive = (1 / weight) * targets[:, j] * loss[:, j]
        loss_negative = (1 / (1 - weight)) * (1 - targets[:, j]) * loss[:, j]
        loss_balanced[:, j] = loss_positive + loss_negative
    loss_balanced.mul_(0.5)  # We have double the loss, so we normalize it back
    return loss_balanced


@torch.no_grad()
def evaluate(loader, network, criterion, classes, target_classes, class_weights=None, device=None):
    device = utils.get_default(device, default=next(network.parameters()).device)
    target_classes_indices = get_class_indices(classes, target_classes)

    network.eval()
    losses = []
    accuracies = []
    for samples, targets in loader:
        samples = samples.to(device)
        targets = targets[:, target_classes_indices].to(device)
        outputs = network(samples)
        loss = criterion(outputs, targets.to(torch.float32))
        if class_weights is not None:
            loss = balance_output(loss, targets, target_classes, class_weights)
        loss = loss.mean()
        accuracy = ((outputs >= 0.0).to(targets) == targets).to(torch.float32)
        if class_weights is not None:
            accuracy = balance_output(accuracy, targets, target_classes, class_weights)
        accuracy = accuracy.mean()
        losses.append(loss.cpu().numpy())
        accuracies.append(accuracy.cpu().numpy())

    return np.mean(losses), np.mean(accuracies)


def classification(network, train_loader, valid_loader, classes, target_classes,
                   train_weights=None, valid_weights=None, i_max=60000, i_print=600, device=None):

    device = utils.get_default(device, default=next(network.parameters()).device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9, nesterov=True,
                          weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i_max // 2, (i_max // 4) * 3],
                                               gamma=0.1)

    target_classes_indices = get_class_indices(classes, target_classes)
    data_augmentation = dutils.Augmentation(flip_horizontal=0.5, flip_vertical=0.0)

    progress = tqdm(total=i_max, position=0)
    i = 0
    while i < i_max:
        for samples, targets in iter(train_loader):
            network.train()
            samples = samples.to(device)
            samples = data_augmentation(samples)
            targets = targets[:, target_classes_indices].to(device)

            outputs = network(samples)
            loss = criterion(outputs, targets.to(torch.float32))
            if train_weights is not None:
                loss = balance_output(loss, targets, target_classes, train_weights)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if (i + 1) % i_print == 0:
                network.eval()
                train_loss, train_accuracy = evaluate(train_loader, network, criterion,
                                                      classes, target_classes,
                                                      class_weights=train_weights, device=device)
                valid_loss, valid_accuracy = evaluate(valid_loader, network, criterion,
                                                      classes, target_classes,
                                                      class_weights=valid_weights, device=device)
                result = f"{('[' + str(i + 1) + ']'):8s}   " \
                         f"Training: {str(train_accuracy * 100):.6}% ({str(train_loss):.6})   " \
                         f"Validation: {str(valid_accuracy * 100):.6}% ({str(valid_loss):.6})"
                progress.write(result)
            i += 1
            progress.update(1)

            if i == i_max:  # Stop at i_max
                break
