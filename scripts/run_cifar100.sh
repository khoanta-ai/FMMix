# ## Base
python main.py --name="CIFAR100_PR18_Base"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --save_path='save_path'

# ## Mixup
python main.py --name="CIFAR100_PR18_ORIGMIXUP"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --mix_alg="mixup" --alpha=1.0 --save_path='save_path'

# ## Manifold Mixup
python main.py --name="CIFAR100_PR18_MANIFOLDMIXUP_RANDLS012"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="mixup_hidden"  --mix_alg="mixup_hidden"  --alpha=2.0 --rand_layers --choice_layers 0 1 2  --save_path='save_path'

python main.py --name="CIFAR100_PR18_MANIFOLDMIXUP_RANDLS0123"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="mixup_hidden"  --mix_alg="mixup_hidden"  --alpha=2.0 --rand_layers --choice_layers 0 1 2 3  --save_path='save_path'

python main.py --name="CIFAR100_PR18_MANIFOLDMIXUP_RANDLS01234"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="mixup_hidden"  --mix_alg="mixup_hidden"  --alpha=2.0 --rand_layers --choice_layers 0 1 2 3 4  --save_path='save_path'


# # ## Cutmix
python main.py --name="CIFAR100_PR18_ORIGCUTMIX"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --mix_alg="cutmix" --alpha=1.0  --cutmix_prob=0.5 --save_path='save_path'


## Mixup + 1234
python main.py --name="CIFAR100_PR18_MIXUP_FMMIX1_RANDLS01234"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_SalMix" --is_fm_mixup  --mix_alg="mixup_fmmix"  --alpha=0.5 --rand_layers --choice_layers 0 1 2 3 4  --save_path='save_path'

# ## Cutmix + 1234
python main.py --name="CIFAR100_PR18_CUTMIX_FMMIX1_RANDLS01234"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_SalMix" --is_fm_mixup  --mix_alg="cutmix_fmmix"  --alpha=0.5 --rand_layers --choice_layers 0 1 2 3 4  --save_path='save_path'
