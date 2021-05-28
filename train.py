import torch
from tqdm.auto import tqdm
import os
import csv
from pprint import pprint
import numpy as np


def write_to_writer(writer, content):
    writer.writerow(content)
    pprint(content)


# refactor by making only 1 writer
def run_epoch(epoch, model, device, optimizer, loader, loss_computer, writer, logger, is_training):
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss, total_margin = 0, 0  # keep track of avg loss, margin in train
    l_correct, g_correct, total = 0, 0, 0  # for validation
    # g = group and l = label
    group_track = {f'g{i}': {'correct_g': 0, 'total': 0, 'margin': 0, 'correct_l': 0} for i in
                   range(4)}  # keeps track of counts of #correct, #total for each group g0-g4
    log_train_every = 200
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(tqdm(loader)):
            batch = tuple(t.to(device) for t in batch)
            x, y, g = batch
            outputs = model(x)

            if is_training:
                optimizer.zero_grad()
                loss = loss_computer(outputs, g)  # we are classifying groups
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (batch_idx % log_train_every) == log_train_every - 1:  # print every 200 mini-batches
                    logger.write('[%d, %5d] loss: %.3f, avg_margin: %.3f' %
                                 (epoch, batch_idx + 1, running_loss / log_train_every, total_margin / total))
                    running_loss = 0.0

            # extra validation
            with torch.no_grad():
                # accuracies and margins
                total += g.size(0)  # total
                _, predicted = torch.max(outputs.data, 1)  # get predictions
                maskg = torch.zeros((len(g), 4), dtype=torch.bool, device=device)
                maskg[np.arange(len(g)), g] = 1
                margins = outputs[maskg] - torch.max(outputs * (~maskg), dim=1)[0]
                total_margin += margins.sum().item()
                label_pred = (predicted // 2 == y)
                for g_idx in range(4):
                    group_track[f'g{g_idx}']['correct_g'] += (
                        (predicted == g)[g == g_idx]).sum().item()  # correctly predict
                    group_track[f'g{g_idx}']['correct_l'] += label_pred[g == g_idx].sum().item()
                    group_track[f'g{g_idx}']['total'] += (
                            g == g_idx).sum().item()  # total number of group's instance encountered
                    group_track[f'g{g_idx}']['margin'] += margins[g == g_idx].sum().item()
                g_correct += (predicted == g).sum().item()
                l_correct += label_pred.sum().item()
        # write stats in dict and csv
        stats_dict = {'epoch': epoch, 'total_acc': f"{l_correct / total:.4f}", 'split_acc': f'{g_correct / total:.4f}',
                      'avg_margin': f"{total_margin / total:.4f}"}
        for g in range(4):
            stats_dict[f'group{g}_acc'] = f"{group_track[f'g{g}']['correct_g'] / group_track[f'g{g}']['total']:.4f}"
            stats_dict[f'total_acc:g{g}'] = f"{group_track[f'g{g}']['correct_l'] / group_track[f'g{g}']['total']:.4f}"
            stats_dict[f'group{g}_margin'] = f"{group_track[f'g{g}']['margin'] / group_track[f'g{g}']['total']:.4f}"
        if is_training:
            stats_dict['loss'] = f'{running_loss / log_train_every:.4f}'

        write_to_writer(writer, stats_dict)


def train(args, model, device, mode, data, logger):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    train_path = open(os.path.join(args.log_dir, 'train.csv'), mode)
    val_path = open(os.path.join(args.log_dir, 'val.csv'), mode)
    test_path = open(os.path.join(args.log_dir, 'test.csv'), mode)

    total_acc_per_group = [f'total_acc:g{i}' for i in range(4)]
    train_columns = ['epoch', 'total_acc', 'group0_acc', 'group1_acc', 'group2_acc', 'group3_acc', 'split_acc', 'loss',
                     'avg_margin', 'group0_margin',
                     'group1_margin', 'group2_margin', 'group3_margin'] + total_acc_per_group

    valtest_columns = ['epoch', 'total_acc', 'group0_acc', 'group1_acc', 'group2_acc', 'group3_acc', 'split_acc',
                       'avg_margin', 'group0_margin',
                       'group1_margin', 'group2_margin', 'group3_margin'] + total_acc_per_group

    train_writer = csv.DictWriter(train_path, fieldnames=train_columns)
    val_writer = csv.DictWriter(val_path, fieldnames=valtest_columns)
    test_writer = csv.DictWriter(test_path, fieldnames=valtest_columns)
    if mode == 'w':
        val_writer.writeheader()
        train_writer.writeheader()
        test_writer.writeheader()

    train_loader, val_loader, test_loader = data['train_loader'], data['val_loader'], data['test_loader']

    for epoch in range(args.resume_from, args.resume_from + args.n_epochs):
        # train
        logger.write(f'Train epoch {epoch}')
        run_epoch(epoch + 1, model, device, optimizer, train_loader, criterion, train_writer, logger, is_training=True)

        # validate
        logger.write(f'Validate epoch {epoch}')
        run_epoch(epoch + 1, model, device, optimizer, val_loader, criterion, val_writer, logger, is_training=False)

        # save
        if (args.save_every is not None) and ((epoch + 1) % args.save_every == 0):
            save_path = os.path.join(args.log_dir, f'model_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)

    # Save model
    save_path = os.path.join(args.log_dir, f'model_{args.resume_from + args.n_epochs}.pth')
    torch.save(model.state_dict(), save_path)

    train_path.close()
    val_path.close()
    test_path.close()
