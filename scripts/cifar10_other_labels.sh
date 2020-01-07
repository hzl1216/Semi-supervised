# 2000, 1000, 500
for sup_size in  2000 1000 500;
do
    python train.py --dataset=cifar10 --optimizer='sgd' --warmup-step=20 --lr=0.03 --weight-decay=0.0005 --epochs=400 --gpu=0  --n-labeled=${sup_size}
    $@
done


