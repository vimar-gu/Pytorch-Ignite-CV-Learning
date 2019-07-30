python -u train.py --model_name=resnet_metric --dataset_name=cars196 --loss_fn=softmax+triplet --use_triplet=1 --bnneck=1 --log_interval=20 --lr=1e-4 --n_epochs=100
