python imagenet.py --warmup 0 --batch_size "{'train':1280,'test':625}" --det_params "{'lr':0.5,'weight_decay':0.0001}" \
                   --gpus 8 --workers 4 --kl_weight "{'kl_min': 1.0, 'kl_max': 1.0, 'last_iter': 8}" \
                   --n_components $2 --prior "{'mean':1.0,'std':0.1}" --sto_params "{'lr':$1,'weight_decay':0.0}" \
                   --root experiments/storesnet50_mixpre_$4_priormean1.0_priorstd0.3_$1_$2_nts$3 \
                   --num_sample "{'train':$3,'test':1}" --posterior "{'mean_init':(1.0,0.50),'std_init':(0.05,0.02)}" \
                   --nodes 1 --nr 0 --seed $4 --use_pretrained --test_freq 1 \
                   --schedule "{'det': [(0.001, 0)], 'sto': []}" --num_epoch 15