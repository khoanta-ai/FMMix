import os
import json
import time
import torch
import copy
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime

from model import ResNet18
from dataloader import load_data
from orig_mixup import mixup_data, mixup_criterion, cutmix_data
from utils import AverageMeter, colorstr, print_batch, prepare_train, accuracy, adjust_learning_rate


import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" #if torch.cuda.is_available() else 'cpu'


def test(args, model, dataloaders, history):
    
    LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                        ('Validation:', 'gpu_mem', 'loss', 'acc'))
        
    phase = 'val'
    criterion = nn.CrossEntropyLoss()
    model.eval()
    fm_mix_flag = False

    print(f"{phase=}, {fm_mix_flag=}")
    _phase = tqdm(dataloaders[phase],
                total=len(dataloaders[phase]),
                bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                unit='batch')

    losses = AverageMeter()
    top1 = AverageMeter()
    total_losses = AverageMeter()
    cls_losses = AverageMeter()
    cls_corrects = AverageMeter()

    for inputs, labels in _phase:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            if args.mix_alg == "mixup":
                outputs, _ = model(inputs, fm_mix=False, target=None)
            elif args.mix_alg == "cutmix":
                outputs, _ = model(inputs, fm_mix=False, target=None)
            elif args.mix_alg == "mixup_hidden":
                outputs, _ = model(inputs, fm_mix=False, target=None, au_layer=None)
            elif args.mix_alg == "fmmix":
                if args.is_fm_mixup:
                    outputs, _ = model(inputs, fm_mix=fm_mix_flag, target=None)
            else:
                outputs, _ = model(inputs, fm_mix=fm_mix_flag, target=None)
            loss = criterion(outputs, labels)
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
        
        print_batch(args, losses, total_losses, cls_losses,
                                    top1, cls_corrects, _phase)
    
    history["val_loss"].append(losses.avg)
    history["val_acc"].append(top1.avg)
    return history
    
    
def train(args, model, optimizer, dataloaders, epoch, history, cur_path):
    
    LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                        ('Training:', 'gpu_mem', 'loss', 'acc'))

    if args.is_fm_mixup:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()
    
    if args.mix_alg == "mixup_hidden":
        softmax = nn.Softmax(dim=1).cuda()
        bce_loss = nn.BCELoss()

    
    current_learning_rate = adjust_learning_rate(optimizer, epoch, args)
    
    phase = "train"
    model.train()
    fm_mix_flag = args.fm_mix_flag


    print(f"{phase=}, {fm_mix_flag=}")
    _phase = tqdm(dataloaders[phase],
                total=len(dataloaders[phase]),
                bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                unit='batch')
    
    losses = AverageMeter()
    top1 = AverageMeter()
    total_losses = AverageMeter()
    cls_losses = AverageMeter()
    cls_corrects = AverageMeter()

    for inputs, labels in _phase:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if args.mix_alg == "mixup":
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.alpha, use_cuda=True)
                outputs, _ = model(inputs, fm_mix=False,  target=None)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            elif args.mix_alg == "cutmix":
                r = np.random.rand(1)
                if args.alpha > 0 and r < args.cutmix_prob:
                    inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, args.alpha, use_cuda=True)
                    outputs, _ = model(inputs, fm_mix=False, target=None)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs, _ = model(inputs, fm_mix=False, target=None)
                    loss = criterion(outputs, labels)

            elif args.mix_alg == "mixup_hidden":
                au_layer = np.random.choice(args.choice_layers)
                outputs, computation_loss_components = model(inputs, au_layer=au_layer, fm_mix=fm_mix_flag)
                loss = bce_loss(softmax(outputs), computation_loss_components)


            elif args.mix_alg == "feature_level":
                if args.is_fm_mixup:
                    outputs, computation_loss_components = model(inputs, target=labels, fm_mix=fm_mix_flag)
                    for comp in range(computation_loss_components[0]):
                        lam_mix = computation_loss_components[1][comp]
                        target_mix = computation_loss_components[2][comp]
                        if comp == 0:
                            loss = criterion(outputs, target_mix) * lam_mix
                        else:
                            loss += criterion(outputs, target_mix) * lam_mix
                    loss = loss.mean()    


            elif args.mix_alg == "fmmix":
                if args.is_fm_mixup:
                    outputs, computation_loss_components = model(inputs, target=labels, fm_mix=fm_mix_flag)
                    for comp in range(computation_loss_components[0]):
                        lam_mix = computation_loss_components[1][comp]
                        target_mix = computation_loss_components[2][comp]
                        if comp == 0:
                            loss = criterion(outputs, target_mix) * lam_mix
                        else:
                            loss += criterion(outputs, target_mix) * lam_mix
                    loss = loss.mean()

                        
            elif args.mix_alg == "mixup_fmmix":
                if args.is_fm_mixup:
                    au_layer = np.random.choice(args.choice_layers)
                    if au_layer == 0:
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, 1.0, use_cuda=True)
                        outputs, computation_loss_components = model(inputs, fm_mix=False, au_layer=au_layer)
                    
                        loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, targets_a, targets_b, lam)
                    else:
                        outputs, computation_loss_components = model(inputs, au_layer=au_layer, target=labels, 
                                                                     fm_mix=fm_mix_flag)
                        
                        if isinstance(computation_loss_components[0], list):
                            num_comp = computation_loss_components[0][0]
                        else:
                            num_comp = computation_loss_components[0]

                        for comp in range(num_comp):
                            lam_mix = computation_loss_components[1][comp]
                            target_mix = computation_loss_components[2][comp]
                            if comp == 0:
                                loss = criterion(outputs, target_mix) * lam_mix
                            else:
                                loss += criterion(outputs, target_mix) * lam_mix
                        loss = loss.mean()


            elif args.mix_alg == "cutmix_fmmix":
                if args.is_fm_mixup:
                    au_layer = np.random.choice(args.choice_layers)
                    if au_layer == 0:
                        inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, 1.0, use_cuda=True)
                        outputs, computation_loss_components = model(inputs, fm_mix=False, au_layer=au_layer)
                        loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, targets_a, targets_b, lam)

                    else:
                    
                        outputs, computation_loss_components = model(inputs, au_layer=au_layer, target=labels,
                                                                     fm_mix=fm_mix_flag)
                        for comp in range(computation_loss_components[0]):
                            lam_mix = computation_loss_components[1][comp]
                            target_mix = computation_loss_components[2][comp]
                            if comp == 0:
                                loss = criterion(outputs, target_mix) * lam_mix
                            else:
                                loss += criterion(outputs, target_mix) * lam_mix
                        loss = loss.mean()

                       
            else:
                outputs, computation_loss_components = model(inputs, fm_mix=fm_mix_flag)
                loss = criterion(outputs, labels)
            
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            loss.backward()
            optimizer.step()
        
        print_batch(args, losses, total_losses, cls_losses, top1, cls_corrects, _phase)

    

    history["train_loss"].append(losses.avg)
    history["train_acc"].append(top1.avg)
    
    
    # Save model for each epoch (overwrite the last model)
    torch.save(model.state_dict(), f"{cur_path}/last.pth")
    print("[INFO] Last model saved")
    return current_learning_rate, history


def run(args, model, optimizer, dataloaders, path):
    since = time.time()
    
    model.to(device)
    
    best_val_acc = 0.0
    train_datetime = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    cur_path = os.path.join(path, train_datetime)
    os.makedirs(cur_path, exist_ok=True)

    history = prepare_train()
    for epoch in range(args.epochs):
        LOGGER.info(colorstr(f'Epoch {epoch}/{args.epochs-1}:'))
        current_learning_rate, history = train(args, model, optimizer, dataloaders, epoch, history, cur_path)

        history = test(args, model, dataloaders, history)

        if history["val_acc"][-1] > best_val_acc:
            best_val_acc = history["val_acc"][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_optim = copy.deepcopy(optimizer.state_dict())
            history['best_epoch'] = epoch
            # Save the best model
            torch.save(model.state_dict(), f"{cur_path}/best.pth")

        # print({"train_loss": history["train_loss"][-1], "train_acc": history["train_acc"][-1],
        #            "val_loss": history["val_loss"][-1], "val_acc": history["val_acc"][-1], 
        #            "current_learning_rate":current_learning_rate, "best_val_acc_line":best_val_acc})
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s with {} epochs'.format(time_elapsed // 3600,time_elapsed % 3600 // 60, time_elapsed % 60, args.epochs))
    print('Best val Acc: {:4f}'.format(best_val_acc))

    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_model_optim)
    history['INFO'] = 'Training complete in {:.0f}h {:.0f}m {:.0f}s with {} epochs -Best Epoch: {} - Best val Acc: {:4f}'.format(time_elapsed // 3600,time_elapsed % 3600 // 60, time_elapsed % 60, args.epochs, history['best_epoch'], best_val_acc)
    print("[INFO] Best model saved")
    
    with open(os.path.join(cur_path, 'result.json'), "w") as outfile:
        json.dump(history, outfile)
    print({"best_epoch":history['best_epoch'], "best_val_acc":best_val_acc})
    
    return model, best_val_acc


def experiment(args):
    args.device = device
    dataloaders = load_data(args, args.batch_size, args.dataset, None, None)

    if args.dataset.lower() == "cifar100":
        num_classes = 100
        stride=1
        args.num_classes = num_classes
    elif args.dataset.lower() == "tinyimagenet200":
        num_classes = 200
        args.num_classes = num_classes
        stride=2
    else:
        raise Exception(f"{args.dataset} dose not know num_classes")

    model = ResNet18(args, num_classes=num_classes, stride=stride).to(device=device)
    
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)
    
    if not args.fm_mix_flag:
        args.p_fm_mix = 0
        print(f"{args.fm_mix_flag=} and {args.p_fm_mix=}")

    print(args)

    name = args.name
    exp_path = os.path.join(args.save_path, name)
    os.makedirs(exp_path, exist_ok=True)
    
    with open(os.path.join(exp_path, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    
    best_mode, best_val_acc = run(args = args,
                                    model = model,
                                    optimizer = optimizer,
                                    dataloaders = dataloaders,
                                    path=exp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='No_Name', type=str, help="Test case name")
    parser.add_argument("--dataset", default='CIFAR100', type=str, help="CIFAR100.")
    parser.add_argument("--model", default='resnet18', type=str, help="Name of model architecure")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--save_path", default="save_path", type=str, help="Path to save model and results")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--decay", default=0.001, type=float)
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')

    
    ## GA augmentation
    parser.add_argument('--fm_mix_flag', action='store_true')  #crossover_flag 
    parser.add_argument("--p_fm_mix", default=None, type=float, help="Agumentation probability") # p_crossover
    parser.add_argument("--fm_augment_name", default=None, type=str, help="FM augment name")

    ### mixup
    parser.add_argument("--alpha", default=None, type=float, help="alpha of mixup.")
    parser.add_argument("--is_fm_mixup", action='store_true')  # is_mixup_ga
    
    ### Random layer
    parser.add_argument("--rand_layers",  action='store_true') 

    ### Orignal Modes
    parser.add_argument("--mix_alg", default='', type=str, help="mixup algorithm")
    parser.add_argument("--cutmix_prob", default=0.0, type=float)
    
    # Test feature map level 
    parser.add_argument("--mix_fm_lv", default=None, type=str, help="Test Mixup or CutMix at feature map level")
    parser.add_argument('--choice_layers', type=int, nargs='+', default=None, help='choice from list layer')
    
    
    args = parser.parse_args()
    
   
    experiment(args)


