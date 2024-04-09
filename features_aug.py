import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import generate_top_k_masks


def fm_mix_level(args, x: torch.Tensor, target: torch.Tensor):

    # Assuming x is your input tensor with shape (B, C, H, W)
    B, C, H, W = x.shape
    
    # Step 1: Shuffle x across B
    shuffle_idx = torch.randperm(B)
    x_shuffle = x[shuffle_idx]

    
    if args.mix_fm_lv == "mixup":
        # Mixup Step 2: Generate constant lambda mask
        masks = torch.ones_like(x)
        lam = torch.tensor(np.random.beta(args.alpha, args.alpha, size=(B, 1, 1, 1)), device=x.device, dtype=x.dtype)
        masks = masks * lam
        # Mixup Step 3: Generate new feature maps
        x = x*(1-masks) + x_shuffle*(masks)

        target_shuffled = target[shuffle_idx]
        p_src = lam.reshape(shape=(B,))
        p_tar = 1 - p_src
        

    elif args.mix_fm_lv == "cutmix":
        lam = args.alpha
        # Cutmix Step 2: Generate  bounding boxes
        lambdas = torch.sqrt(lam * torch.rand(B, 1, 1, 1, device=x.device))
        
        # Cutmix Step 3: Create masks
        Wb = (lambdas * W).squeeze(-1)
        Hb = (lambdas * H).squeeze(-1)
        x_range = torch.arange(W, device=x.device)[None, :].expand(B, C, H, W)
        y_range = torch.arange(H, device=x.device)[:, None].expand(B, C, H, W)

        bcx = torch.randint(0, W, size=(B, 1, 1, 1), device=x.device)
        bcy = torch.randint(0, H, size=(B, 1, 1, 1), device=x.device)

        x_min = torch.clamp(bcx - Wb[..., None] // 2, 0, W)
        y_min = torch.clamp(bcy - Hb[..., None] // 2, 0, H)

        x_max = torch.clamp(x_min + Wb[..., None], 0, W)
        y_max = torch.clamp(y_min + Hb[..., None], 0, H)

        masks = (x_range >= x_min) & (x_range < x_max) & (y_range >= y_min) & (y_range < y_max)
        
        # CutMix Step 4: Generate new feature maps
        x = x*~masks + x_shuffle*masks

        target_shuffled = target[shuffle_idx]
        p_src = torch.sum(masks, dim=[1,2,3])/(C*W*H)
        p_tar = 1 - p_src


    else:
        lam = args.alpha
        raise Exception("Not implemented fm_mix_level")


    computation_loss_components = [
        2,
        [p_tar, p_src],
        [target, target_shuffled]
    ]

    return x, None, computation_loss_components


def ffmix2(args, x: torch.Tensor, target: torch.Tensor):
    # Assuming x is your input tensor with shape (B, C, H, W)
    B, C, H, W = x.shape
    lam = args.alpha
    # Step 1: Shuffle x across B
    shuffle_idx = torch.randperm(B)
    x_shuffle = x[shuffle_idx]

    # Step 2: Find the highest value in each feature map of x_shuffle
    max_val_indices = torch.argmax(x_shuffle.view(B, C, -1), dim=2) # Indices of max values
    max_indices_unraveled = torch.stack((max_val_indices // W, max_val_indices % W), dim=-1)

    # Step 3: Generate bounding boxes
    lambdas = torch.sqrt(lam * torch.rand(B, 1, 1, 1, device=x.device))
    m = lambdas.max()
    m2 = lambdas.min()
    Wb = (lambdas * W).squeeze(-1)
    Hb = (lambdas * H).squeeze(-1)

    # Step 4: Create masks
    x_range = torch.arange(W, device=x.device)[None, :].expand(B, C, H, W)
    y_range = torch.arange(H, device=x.device)[:, None].expand(B, C, H, W)

    x_min = torch.clamp(max_indices_unraveled[..., 1][..., None, None] - Wb[..., None] // 2, 0, W)
    y_min = torch.clamp(max_indices_unraveled[..., 0][..., None, None] - Hb[..., None] // 2, 0, H)

    x_max = torch.clamp(x_min + Wb[..., None], 0, W)
    y_max = torch.clamp(y_min + Hb[..., None], 0, H)

    masks = (x_range >= x_min) & (x_range < x_max) & (y_range >= y_min) & (y_range < y_max)

    ###########################################################
    shift_tar = x - x.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    per_SEM_tar = shift_tar / torch.sum(shift_tar, dim=[2,3], keepdim=True)
    p_tar = torch.sum(per_SEM_tar*(1-masks.float()), dim=[1,2,3])/C
    m_tar = p_tar.max()
    shift_src = x_shuffle - x_shuffle.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    per_SEM_src = shift_src / torch.sum(shift_src, dim=[2,3], keepdim=True)
    p_src = torch.sum(per_SEM_src*masks.float(), dim=[1,2,3])/C
    m_src = p_src.max()
    # Step 5: Extract and paste patches
    # patches = x_shuffle * masks
    # x[masks] = patches[masks]
    x = x*(1-masks.float()) + x_shuffle*masks
    
    target_shuffled = target[shuffle_idx]
    
    computation_loss_components = [
        2,
        [p_tar, p_src],
        [target, target_shuffled]
    ]

    return x, computation_loss_components


def ffmix1(args, x: torch.Tensor, target: torch.Tensor):
    # Assuming x is your input tensor with shape (B, C, H, W)
    B, C, H, W = x.shape
    lam = args.alpha
    # Step 1: Shuffle x across B
    shuffle_idx = torch.randperm(B)
    x_shuffle = x[shuffle_idx]

    # Step 2: Find the highest value in each feature map of x_shuffle
    max_val_indices = torch.argmax(x_shuffle.view(B, C, -1), dim=2) # Indices of max values
    max_indices_unraveled = torch.stack((max_val_indices // W, max_val_indices % W), dim=-1)

    # Step 3: Generate bounding boxes
    lambdas = torch.sqrt(lam * torch.rand(B, 1, 1, 1, device=x.device))
    # lambdas = 0.0088*torch.ones((B,C,1,1), device=x.device, dtype=x.dtype)
    m1 = lambdas.max()
    m2 = lambdas.min()
    Wb = (lambdas * W).squeeze(-1)
    Hb = (lambdas * H).squeeze(-1)

    # Step 4: Create masks
    x_range = torch.arange(W, device=x.device)[None, :].expand(B, C, H, W)
    y_range = torch.arange(H, device=x.device)[:, None].expand(B, C, H, W)

    x_min = torch.clamp(max_indices_unraveled[..., 1][..., None, None] - Wb[..., None] // 2, 0, W)
    y_min = torch.clamp(max_indices_unraveled[..., 0][..., None, None] - Hb[..., None] // 2, 0, H)

    x_max = torch.clamp(x_min + Wb[..., None], 0, W)
    y_max = torch.clamp(y_min + Hb[..., None], 0, H)

    masks = (x_range >= x_min) & (x_range < x_max) & (y_range >= y_min) & (y_range < y_max)

    # Step 5: Extract and paste patches
    # patches = x_shuffle * masks
    # x[masks] = patches[masks]
    x = x*~masks + x_shuffle*masks

    target_shuffled = target[shuffle_idx]
    p_src = torch.sum(masks, dim=[1,2,3])/(C*W*H)
    p_tar = 1 - p_src
    m_tar = p_tar.max()
    m_tar_m = p_tar.min()
    m_src = p_src.max()
    m_src_m = p_src.min()
    
    computation_loss_components = [
        2,
        [p_tar, p_src],
        [target, target_shuffled]
    ]

    return x, computation_loss_components


def get_ks_maxft(x_size):
    if x_size == 2:
        ks = 1 
        maxft = 2
    else:
        ks = np.array([i for i in range(2, x_size) if x_size%i==0 ])
        if len(ks) == 1:
            ks = ks[0]
        else:
            ks =ks[-2]
        total_patches = (x_size/ks) * (x_size/ks)
        total_points = total_patches*ks*ks
        maxft = total_patches/2
    return int(ks), int(maxft)


def ffmix3(args, x:torch.Tensor, target:torch.Tensor):
    B, C, H, W= x.size()
    kernel_size, max_features = get_ks_maxft(W)
    k_max_features = torch.randint(low=0, high=max_features+1, size=[])
    if k_max_features == 0:
        attention_mask = torch.zeros_like(x)
    else:
        attention_mask = generate_top_k_masks(x, k_max_features, kernel_size, is_min=args.is_min)
    
    indices = torch.randperm(x.size(0))
    attention_mask = attention_mask[indices]
    x = x*(1-attention_mask) + x[indices]*attention_mask
    
    lam = torch.sum(attention_mask, dim=[1,2,3])/(C*W*H)
    # target_reweighted = target_reweighted * (1-lam) + target_shuffled_onehot * lam
    target_shuffled = target[indices]
    computation_loss_components = [
        2,
        [1-lam, lam],
        [target, target_shuffled]
    ]
    return x, computation_loss_components


def ffmix4(args, x:torch.Tensor, target:torch.Tensor):
    B, C, H, W= x.size()
    kernel_size, max_features = get_ks_maxft(W)
    k_max_features = torch.randint(low=0, high=max_features+1, size=[])
    # k_max_features = torch.tensor(8, dtype=torch.int64 )
    if k_max_features == 0:
        attention_mask = torch.zeros_like(x)
    else:
        attention_mask = generate_top_k_masks(x, k_max_features, kernel_size, is_min=args.is_min)
    indices = torch.randperm(x.size(0))
    attention_mask = attention_mask[indices]
    shift_tar = x - x.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    per_SEM_tar = shift_tar / torch.sum(shift_tar, dim=[2,3], keepdim=True)
    p_tar = torch.sum(per_SEM_tar*(1-attention_mask), dim=[1,2,3])/C

    shift_src = x[indices] - x[indices].min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    per_SEM_src = shift_src / torch.sum(shift_src, dim=[2,3], keepdim=True)
    p_src = torch.sum(per_SEM_src*attention_mask, dim=[1,2,3])/C
    x = x*(1-attention_mask) + x[indices]*attention_mask
    m = p_src.max()
    mi = p_src.min()
    # lam = torch.sum(attention_mask, dim=[1,2,3])/(C*W*H)
    # target_reweighted = target_reweighted * (1-lam) + target_shuffled_onehot * lam
    target_shuffled = target[indices]
    computation_loss_components = [
        2,
        [p_tar, p_src],
        [target, target_shuffled]
    ]
    return x, computation_loss_components

