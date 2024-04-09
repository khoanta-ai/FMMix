## top k grid - perSem ratio - specific layer 1
python main.py --name="CIFAR100_PR18_FMMIX4_SP1"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 1  --save_path='save_path'

## top k grid - perSem ratio - specific layer 2
python main.py --name="CIFAR100_PR18_FMMIX4_SP2"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 2  --save_path='save_path'

## top k grid - perSem ratio - specific layer 3
python main.py --name="CIFAR100_PR18_FMMIX4_SP3"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 3  --save_path='save_path'

## top k grid - perSem ratio - specific layer 4
python main.py --name="CIFAR100_PR18_FMMIX4_SP4"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 4 --save_path='save_path'


## top k grid - perSem ratio - Combination 2 layers
python main.py --name="CIFAR100_PR18_FMMIX4_RANDLS12"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 1 2  --save_path='save_path'

python main.py --name="CIFAR100_PR18_FMMIX4_RANDLS23"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 2 3  --save_path='save_path'

python main.py --name="CIFAR100_PR18_FMMIX4_RANDLS34"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 3 4  --save_path='save_path'


## top k grid - perSem ratio - Combination 3 layers
python main.py --name="CIFAR100_PR18_FMMIX4_RANDLS123"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 1 2 3  --save_path='save_path'

python main.py --name="CIFAR100_PR18_FMMIX4_RANDLS234"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 2 3 4  --save_path='save_path'

# top k grid - perSem ratio - Combination 4 layers
python main.py --name="CIFAR100_PR18_FMMIX4_RANDLS1234"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="ffmix4"  --is_fm_mixup  --mix_alg="fmmix"  --alpha=0.5 --rand_layers --choice_layers 1 2 3 4  --save_path='save_path'





















