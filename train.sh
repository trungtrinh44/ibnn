python imagenet.py --gpus 4 --root experiments/storesnet50_$4_100e_posteriormeaninit1.0_posteriorstdinit0.01_priormean1.0_priorstd0.03_$1_$2_nts$3_kl60_wd0.0001_stolranneal \
                   --warmup 1 --batch_size "{'train':256,'test':400}" \
                   --det_params "{'lr':0.10,'weight_decay':1e-4}" \
                   --workers 4 --kl_weight "{'kl_min': 0.0, 'kl_max': 1.0, 'last_iter': 60}" \
                   --n_components $2 --prior "{'mean':1.0,'std':0.03}" --sto_params "{'lr':$1,'weight_decay':0.0}" \
                   --num_sample "{'train':$3,'test':1}" --posterior "{'mean_init':(1.0,0.1),'std_init':(0.01,0.002)}" \
                   --seed $4 --test_freq 1 --sgd_params "{'momentum': 0.9, 'nesterov': True}"\
                   --schedule "{'det': [(0.1, 30), (0.01, 60), (0.001, 90)], 'sto': [(0.1, 30), (0.01, 60)]}" --num_epoch 100
