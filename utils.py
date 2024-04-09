import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prepare_train():
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [],
               "train_att_loss": [], "train_att_acc":[], "train_total_loss":[],
               "train_cls_loss": [], "train_cls_acc":[],
               "val_att_loss": [], "val_att_acc":[], "val_total_loss":[],
               "train_mask_loss": [], "train_mask_acc":[],
               "val_mask_loss": [], "val_mask_acc":[],
               "val_cls_loss": [], "val_cls_acc":[]}
    return history


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def print_batch(args, losses, total_losses, cls_losses,
                                top1, cls_corrects, _phase):
    # one batch
    if total_losses.count > 0:
        if args.multipath:
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
            desc = ('%35s' + '%15.6g' * 5) % (mem, losses.avg, top1.avg,
                                             cls_losses.avg, cls_corrects.avg,
                                             total_losses.avg)
    else:
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
        desc = ('%35s' + '%15.6g' * 2) % (mem, losses.avg, top1.avg)
    _phase.set_description_str(desc)


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']



def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
    return y_onehot


def get_attention_mask(x:torch.Tensor, attention_map:torch.Tensor, 
             k:int, is_min:bool=False, is_top:bool=False, grid_size:int=None, threshold:int=None):
    B, C, H, W = x.size()
    
    # Average pooling
    avg_pool = nn.AvgPool2d(kernel_size=grid_size, stride=grid_size)
    # Perform average pooling
    averaged = avg_pool(attention_map)
    
    # Flatten the pooled map and find top k values
    flattened = averaged.reshape(B, -1)
    
    if is_top: # get top k values
        if is_min:
            averaged *= -1
        num_girds = (H * W) // grid_size
        assert k <= num_girds, f"{k= } > number of grids= {num_girds}"
        topk_values, k_indices = torch.topk(flattened, k, dim=1)
        # Create mask
        mask_small = torch.zeros_like(flattened).scatter_(1, k_indices, 1)
        mask_small = mask_small.view_as(averaged)
    else: # get randomly k values 
        if is_min:
            flattened_mask = flattened < threshold
        else:
            flattened_mask = flattened > threshold
        flattened_mask = flattened_mask.int()
        # Sort each row in descending order and keep track of the original indices
        sorted_mask, sorted_indices = flattened_mask.sort(dim=1, descending=True)
        # Create a cutoff mask with k ones followed by zeros in each row
        cutoff_mask = torch.cat([torch.ones(B, k), torch.zeros(B, flattened.size(1) - k)], dim=1).to(x.device).int()
        # Apply the cutoff mask
        reduced_mask = sorted_mask * cutoff_mask
        _, original_indices = sorted_indices.sort(dim=1)
        mask_small = reduced_mask.gather(1, original_indices)
        mask_small = mask_small.view_as(averaged)
    if grid_size > 1: # grid
        # Upscale mask
        attention_mask = mask_small.repeat_interleave(grid_size, dim=2).repeat_interleave(grid_size, dim=3)
    else: # pixel
        attention_mask = mask_small
    return attention_mask



def generate_top_k_masks(feature_map, k, patch_size, is_min=False):
    """
    Generate masks for top k patches in each channel of a feature map without using for-loops.

    Parameters:
    - feature_map (torch.Tensor): The input feature map with shape (B, C, H, W).
    - k (int): The number of top patches to select in each channel.
    - patch_size (int): The size of each patch (assuming square patches).

    Returns:
    - torch.Tensor: A mask tensor with the same shape as feature map, 
                    containing 1s in positions belonging to top k patches in each channel.
    """
    # Average pooling to reduce to patch size
    pooled = F.avg_pool2d(feature_map, kernel_size=patch_size, stride=patch_size)

    # Flatten the pooled feature map except for the batch and channel dimensions
    B, C, H, W = pooled.shape
    flattened = pooled.view(B, C, -1)
    if is_min:
        flattened *= -1
    # Get indices of top k elements
    topk_indices = flattened.topk(k, dim=2).indices

    # Create a range tensor to match the batch and channel dimensions
    batch_range = torch.arange(B)[:, None, None]
    channel_range = torch.arange(C)[None, :, None]

    # Use broadcasting to expand ranges to match the shape of topk_indices
    batch_range = batch_range.expand(B, C, k)
    channel_range = channel_range.expand(B, C, k)

    # Initialize the mask with zeros
    mask = torch.zeros_like(flattened, dtype=torch.float32)

    # Set the top k positions to 1 in the mask using advanced indexing
    mask[batch_range, channel_range, topk_indices] = 1

    # Reshape the mask to the original pooled feature map's shape
    mask = mask.view(B, C, H, W)
    # Upsample the mask to the original feature map's size
    mask = F.interpolate(mask, scale_factor=patch_size, mode='nearest')
    return (mask > 0).float()


def adjust_learning_rate(optimizer, epoch, args):
    gammas = args.gammas
    schedule = args.schedule
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

