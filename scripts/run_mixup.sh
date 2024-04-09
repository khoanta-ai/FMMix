# Mixup Feature level - Specific Layer
python main.py --name="CIFAR100_PR18_MIXUP_SP1"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level" --mix_fm_lv='mixup'  --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 1  --save_path='save_path'

python main.py --name="CIFAR100_PR18_MIXUP_SP2"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level"  --mix_fm_lv='mixup' --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 2  --save_path='save_path'

python main.py --name="CIFAR100_PR18_MIXUP_SP3"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level"  --mix_fm_lv='mixup' --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 3  --save_path='save_path'

python main.py --name="CIFAR100_PR18_MIXUP_SP4"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level"  --mix_fm_lv='mixup' --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 4  --save_path='save_path'


# Mixup Feature level - Combination 2 layers
python main.py --name="CIFAR100_PR18_MIXUP_RANDLS12"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level" --mix_fm_lv='mixup'  --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 1 2  --save_path='save_path'

python main.py --name="CIFAR100_PR18_MIXUP_RANDLS23"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level" --mix_fm_lv='mixup'  --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 2 3  --save_path='save_path'

python main.py --name="CIFAR100_PR18_MIXUP_RANDLS34"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level" --mix_fm_lv='mixup'  --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 3 4  --save_path='save_path'


# Mixup Feature level - Combination 3 layers
python main.py --name="CIFAR100_PR18_MIXUP_RANDLS123"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level" --mix_fm_lv='mixup'  --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 1 2 3  --save_path='save_path'

python main.py --name="CIFAR100_PR18_MIXUP_RANDLS234"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level" --mix_fm_lv='mixup'  --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 2 3 4  --save_path='save_path'


# Mixup Feature level - Combination 4 layers
python main.py --name="CIFAR100_PR18_MIXUP_RANDLS1234"  --dataset='CIFAR100'  --model='resnet18'  --lr=0.1  --momentum=0.9 --decay=0.0001 --gammas 0.1 0.1 --schedule 100 150  --batch_size=128  --epochs=200 --fm_mix_flag  --p_fm_mix=1.0  --fm_augment_name="fm_mix_level" --mix_fm_lv='mixup'  --is_fm_mixup  --mix_alg="feature_level"  --alpha=1.0 --rand_layers --choice_layers 1 2 3 4  --save_path='save_path'