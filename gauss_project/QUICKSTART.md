# GAUSS 项目快速启动指南

## 1. 安装依赖

```bash
cd gauss_project
pip install -r requirements.txt
```

## 2. 运行示例

### 2.1 快速示例
```bash
python example.py
```

### 2.2 训练单个数据集
```bash
# 训练同质图数据集 (Cora)
python train.py --dataset Cora --epochs 500

# 训练异质图数据集 (Chameleon)
python train.py --dataset Chameleon --num-blocks 4 --epochs 500
```

### 2.3 运行完整实验 (10次运行)
```bash
# Cora数据集
python run_experiments.py --dataset Cora --num-runs 10

# Chameleon数据集
python run_experiments.py --dataset Chameleon --num-runs 10 --num-blocks 4
```

## 3. 支持的数据集

### 同质图数据集
- Cora, CiteSeer, PubMed
- WikiCS
- Computers, Photo

### 异质图数据集
- Chameleon, Squirrel
- Actor
- Cornell, Texas, Wisconsin

## 4. 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --dataset | 数据集名称 | Cora |
| --num-blocks | 块的数量 k | 3 |
| --lambda-param | λ 参数 | 10.0 |
| --gamma | γ 参数 | 1.0 |
| --hidden-dim | 隐藏层维度 | 256 |
| --lr | 学习率 | 0.01 |
| --dropout | Dropout率 | 0.5 |
| --epochs | 训练轮数 | 500 |

## 5. 预期结果

### 同质图数据集
- Cora: ~84.31% ± 1.63%
- CiteSeer: ~73.14% ± 0.52%
- PubMed: ~86.23% ± 0.28%

### 异质图数据集
- Chameleon: ~76.89% ± 1.87%
- Squirrel: ~67.93% ± 1.40%
- Actor: ~37.37% ± 0.76%

## 6. 常见问题

### Q1: CUDA内存不足
**解决方案**: 降低 `--hidden-dim` 参数

```bash
python train.py --dataset Cora --hidden-dim 128
```

### Q2: 训练速度慢
**解决方案**: 减少 `--max-iter` 参数

```bash
python train.py --dataset Cora --max-iter 10
```

### Q3: 性能不佳
**解决方案**: 根据数据集类型调整 `--num-blocks`
- 同质图: 使用较小的值 (2-3)
- 异质图: 使用较大的值 (3-5)

## 7. 项目结构

```
gauss_project/
├── models/           # GAUSS模型实现
├── utils/            # 数据加载和工具函数
├── configs/          # 配置文件
├── train.py          # 训练脚本
├── run_experiments.py # 实验脚本
├── example.py        # 快速示例
└── README.md         # 详细文档
```

## 8. 论文信息

**标题**: GAUSS: GrAph-customized Universal Self-Supervised Learning  
**会议**: WWW 2024  
**作者**: Liang Yang et al.

## 9. 核心创新点

1. **局部自适应传播**: 使用局部可学习的传播矩阵替代全局参数
2. **k-块对角正则化**: 通过拉普拉斯矩阵的最小k个特征值实现
3. **无需标签**: 完全自监督学习,不依赖节点标签
4. **通用性强**: 同时适用于同质图和异质图

## 10. 下一步

- 阅读完整的 README.md 文档
- 查看 configs/config.yaml 中的详细配置
- 尝试不同的参数组合
- 在自己的数据集上测试

---
如有问题,请参考 README.md 或提交 Issue。
