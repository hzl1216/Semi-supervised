# 4000 2000, 1000, 500
for sup_size in 4000 2000 1000 500;
do
    python cifar10.py --optimizer='sgd' --warmup-step=20 --lr=0.03 --weight-decay=0.0005 --epochs=400 --gpu=0  --n-labeled=${sup_size}
    $@
done

# 250
python cifar10.py --epochs=150 --warmup-step=15 --optimizer='sgd' --consistency-weight=6.0 --lr=0.03 --weight-decay=0.0007 --scheduler=exp --n-labeled=250 --gpu=1 --batch-size=16 --unsup-ratio=20
