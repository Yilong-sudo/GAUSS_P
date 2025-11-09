#!/bin/bash
# 运行脚本示例

# 1. 快速测试（验证安装）
echo "Running quick test..."
python test.py

# 2. 在Cora数据集上训练（同质图）
echo "Training on Cora..."
python train.py --dataset Cora --epochs 500 --num-blocks 2

# 3. 在Chameleon数据集上训练（异质图）
echo "Training on Chameleon..."
python train.py --dataset Chameleon --epochs 500 --num-blocks 4

# 4. 运行完整实验（10次运行）
echo "Running full experiments on Cora..."
python run_experiments.py --dataset Cora --num-runs 10 --num-blocks 2

# 5. 所有数据集的批量实验
for dataset in Cora CiteSeer PubMed Chameleon Squirrel Actor
do
    echo "Running experiments on $dataset..."
    python run_experiments.py --dataset $dataset --num-runs 10
done
