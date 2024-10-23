import os 
import time
import json
import torch
import argparse
from model import ResNet18, ResNet50 
from utils import prepare_train
from dataloader import load_data
from main import test

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else 'cpu'
model_dict = {
    "resnet18": ResNet18,
    "resnet50": ResNet50,
}

def eval(args, model, dataloaders):
    since = time.time()
    model.to(device)
    history = prepare_train()
    history = test(args, model, dataloaders, history)  
    time_elapsed = time.time() - since
    best_val_acc = history["val_acc"][-1]
    print('Best val Acc: {:4f}'.format(best_val_acc))
    
    history['INFO'] = 'Training complete in {:.0f}h {:.0f}m {:.0f}s with {}  - Best val Acc: {:4f}'.format(time_elapsed // 3600,time_elapsed % 3600 // 60, time_elapsed % 60, args.epochs,  best_val_acc)
    
    return model, best_val_acc


def experiment(args):
    args.device = device
    dataloaders = load_data(args, args.batch_size, args.dataset)

    if args.dataset.lower() == "cifar100":
        num_classes = 100
        stride=1
    elif args.dataset.lower() == "tinyimagenet200":
        num_classes = 200
        stride=2
    else:
        raise Exception(f"{args.dataset} dose not know num_classes")

    if not args.fm_mix_flag:
        args.p_fm_mix = 0
        print(f"{args.fm_mix_flag=} and {args.p_fm_mix=}")

    args.num_classes = num_classes
    
    model = model_dict[args.model](args, num_classes=num_classes, stride=stride)
    model.load_state_dict(torch.load(args.weight_path))
    model = model.to(device=device)
    
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)
    
    

    print(args)

    name = args.name
    exp_path = os.path.join(args.save_path, name)
    os.makedirs(exp_path, exist_ok=True)
    
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

   
    best_mode, best_val_acc = eval(args = args,
                                    model = model,
                                    dataloaders = dataloaders) 



if __name__=="__main__":
    par = argparse.ArgumentParser()
    par.add_argument("--config_path", default='', type=str, help="config path")
    par.add_argument("--weight_path", default='', type=str, help="checkpoint path")
    ar = par.parse_args()
    parser = argparse.ArgumentParser()
    with open(ar.config_path, 'rt') as f:
        args = argparse.Namespace()
        args.__dict__.update(json.load(f))
    args.weight_path = ar.weight_path
    print(args)
    experiment(args)
    
    