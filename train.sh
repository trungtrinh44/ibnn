python multi_gpu_train.py --seed 132 --model StoVGG16 --kl_weight "{'kl_min':0.0,'kl_max':1.0,'last_iter':200}" \
            --batch_size "{'train':128,'test':100}" --root experiments/StoVGG16_cifar100 --prior "{'mean':1.0,'std':0.3}" \
            --n_components 48 --det_params "{'lr':0.05,'weight_decay':0.0003}" --sto_params "{'lr':9.6,'weight_decay':0.0}" \
            --sgd_params "{'momentum':0.9,'nesterov':True,'dampening':0.0}" --num_sample "{'train':6,'test':1}" \
            --dataset vgg_cifar100 --lr_ratio "{'det':0.01,'sto':1.0}" \
            --posterior "{'mean_init':(1.0,0.75),'std_init':(0.05,0.02)}" --milestones 0.5 0.9 \
            --nodes 1 --gpus 2 --nr 0