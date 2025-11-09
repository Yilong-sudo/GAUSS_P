# Windows批处理脚本示例

# 1. 快速测试（验证安装）
Write-Host "Running quick test..." -ForegroundColor Green
python test.py

# 2. 在Cora数据集上训练（同质图）
Write-Host "`nTraining on Cora..." -ForegroundColor Green
python train.py --dataset Cora --epochs 500 --num-blocks 2

# 3. 在Chameleon数据集上训练（异质图）
Write-Host "`nTraining on Chameleon..." -ForegroundColor Green
python train.py --dataset Chameleon --epochs 500 --num-blocks 4

# 4. 运行完整实验（10次运行）
Write-Host "`nRunning full experiments on Cora..." -ForegroundColor Green
python run_experiments.py --dataset Cora --num-runs 10 --num-blocks 2

# 5. 测试所有数据集
$datasets = @("Cora", "CiteSeer", "PubMed", "Chameleon", "Squirrel", "Actor")
foreach ($dataset in $datasets) {
    Write-Host "`nRunning experiments on $dataset..." -ForegroundColor Green
    python run_experiments.py --dataset $dataset --num-runs 10
}
