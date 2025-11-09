# GAUSS: GrAph-customized Universal Self-Supervised Learning

PyTorch implementation of **GAUSS** from the paper:

> **GAUSS: GrAph-customized Universal Self-Supervised Learning**  
> Liang Yang et al.  
> WWW 2024  
> [Paper Link](https://doi.org/10.1145/3589334.3645453)

## 📋 Overview

GAUSS is a novel graph-customized universal self-supervised learning method that addresses the challenge of learning universal graph representations without labels. Unlike existing methods that employ global parameters, GAUSS uses locally learnable propagation by exploiting local attribute distribution.

### Key Features

- ✅ **Universal**: Works on both homophilic and heterophilic graphs
- ✅ **Self-Supervised**: No need for labeled data during training
- ✅ **Graph-Customized**: Learns graph-specific propagation patterns
- ✅ **Robust**: Resistant to noise in both features and edges
- ✅ **Efficient**: Uses k-block diagonal regularization for structure learning

## 🔧 Installation

### Requirements

- Python >= 3.7
- PyTorch >= 1.10.0
- PyTorch Geometric >= 2.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0

### Install from Source

```bash
# Clone the repository
cd gauss_project

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## 🚀 Quick Start

### Training on a Single Dataset

```bash
# Train on Cora (homophilic dataset)
python train.py --dataset Cora --epochs 500

# Train on Chameleon (heterophilic dataset)
python train.py --dataset Chameleon --num-blocks 4 --epochs 500
```

### Running Multiple Experiments

```bash
# Run 10 experiments with different seeds
python run_experiments.py --dataset Cora --num-runs 10

# Run experiments on heterophilic dataset
python run_experiments.py --dataset Chameleon --num-runs 10 --num-blocks 4
```

## 📊 Supported Datasets

### Homophilic Datasets
- **Cora**: Citation network (2,708 nodes, 7 classes)
- **CiteSeer**: Citation network (3,327 nodes, 6 classes)
- **PubMed**: Citation network (19,717 nodes, 3 classes)
- **WikiCS**: Wikipedia network (11,701 nodes, 10 classes)
- **Computers**: Amazon co-purchase (13,752 nodes, 10 classes)
- **Photo**: Amazon co-purchase (7,650 nodes, 8 classes)

### Heterophilic Datasets
- **Chameleon**: Wikipedia network (2,277 nodes, 5 classes)
- **Squirrel**: Wikipedia network (5,201 nodes, 5 classes)
- **Actor**: Actor co-occurrence (7,600 nodes, 5 classes)
- **Cornell**: WebKB network (183 nodes, 5 classes)
- **Texas**: WebKB network (183 nodes, 5 classes)
- **Wisconsin**: WebKB network (251 nodes, 5 classes)

## 🎯 Usage Examples

### Basic Training

```python
from models import GAUSS
from utils import DataLoader

# Load data
data_loader = DataLoader(root='./data')
data, num_features, num_classes = data_loader.get_data('Cora')

# Create model
model = GAUSS(
    in_features=num_features,
    hidden_features=256,
    out_features=num_classes,
    num_blocks=3,
    lambda_param=10.0,
    gamma=1.0
)

# Training loop (see train.py for complete example)
```

### Custom Configuration

```bash
python train.py \
    --dataset Cora \
    --hidden-dim 256 \
    --num-blocks 3 \
    --lambda-param 10.0 \
    --gamma 1.0 \
    --lr 0.01 \
    --dropout 0.5 \
    --epochs 500
```

## ⚙️ Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `num_blocks` | Number of blocks (k) for k-block diagonal regularization | 3 | 1-10 |
| `lambda_param` | Balance parameter between Z and B | 10.0 | 1-100 |
| `gamma` | Block diagonal regularization strength | 1.0 | 0.1-50 |
| `max_iter` | Maximum iterations for optimization | 20 | 10-50 |
| `hidden_dim` | Hidden layer dimension | 256 | 64-512 |
| `dropout` | Dropout rate | 0.5 | 0.0-0.8 |
| `lr` | Learning rate | 0.01 | 0.001-0.1 |

### Parameter Tuning Guide

- **For homophilic graphs**: Use smaller `num_blocks` (2-3)
- **For heterophilic graphs**: Use larger `num_blocks` (3-5)
- **For noisy data**: Increase `gamma` for stronger regularization
- **For small datasets**: Reduce `hidden_dim` and increase `dropout`

## 📈 Results

Expected performance (mean accuracy ± std over 10 runs):

### Homophilic Datasets
| Dataset | GAUSS | GCN | GRACE |
|---------|-------|-----|-------|
| Cora | 84.31±1.63 | 82.32±1.79 | 83.30±0.40 |
| CiteSeer | 73.14±0.52 | 72.13±1.17 | 71.41±0.38 |
| PubMed | 86.23±0.28 | 84.90±0.38 | 86.70±0.34 |

### Heterophilic Datasets
| Dataset | GAUSS | GCN | FAGCN |
|---------|-------|-----|-------|
| Chameleon | 76.89±1.87 | 59.63±2.32 | 63.44±2.05 |
| Squirrel | 67.93±1.40 | 36.28±1.52 | 41.17±1.94 |
| Actor | 37.37±0.76 | 30.83±0.77 | 36.81±0.26 |

## 🏗️ Project Structure

```
gauss_project/
├── models/
│   ├── __init__.py
│   └── gauss.py          # GAUSS model implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # Data loading utilities
│   └── utils.py          # Helper functions
├── configs/
│   └── config.yaml       # Configuration file
├── data/                 # Dataset directory (auto-created)
├── checkpoints/          # Model checkpoints (auto-created)
├── results/              # Experiment results (auto-created)
├── train.py              # Training script
├── run_experiments.py    # Multi-run experiment script
├── requirements.txt      # Dependencies
├── setup.py              # Setup file
└── README.md             # This file
```

## 🔬 Algorithm Details

GAUSS learns affinity matrices for ego-networks using self-representative learning with k-block diagonal regularization:

```
min_{Z,B,W} (1/2)||X - XZ||² + (λ/2)||Z - B||² + γ⟨Diag(B1) - B, W⟩

s.t. diag(B) = 0, B ≥ 0, B = B^T
     0 ⪯ W ⪯ I, Tr(W) = k
```

The optimization uses alternating minimization (ADMM) with:
1. **W-update**: Compute k smallest eigenvectors of Laplacian
2. **Z-update**: Solve linear system
3. **B-update**: Projection onto constraint set

## 🛠️ Advanced Usage

### Adding Noise for Robustness Testing

```python
from utils import DataLoader

data_loader = DataLoader()
data, _, _ = data_loader.get_data('Cora')

# Add 10% feature noise
data = DataLoader.add_noise_to_features(data, noise_ratio=0.1)

# Add 10% edge noise
data = DataLoader.add_noise_to_edges(data, noise_ratio=0.1)
```

### Computing Homophily Ratio

```python
homophily = data_loader.compute_homophily_ratio(data)
print(f"Homophily ratio: {homophily:.4f}")
```

### Extracting Embeddings

```python
# Get node embeddings without classification
embeddings = model.get_embeddings(data.x, data.edge_index)
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yang2024gauss,
  title={GAUSS: GrAph-customized Universal Self-Supervised Learning},
  author={Yang, Liang and Hu, Weixiao and Xu, Jizhong and Shi, Runjie and He, Dongxiao and Wang, Chuan and Cao, Xiaochun and Wang, Zhen and Niu, Bingxin and Guo, Yuanfang},
  booktitle={Proceedings of the ACM Web Conference 2024},
  pages={XXX--XXX},
  year={2024},
  organization={ACM}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

This implementation is based on the paper:
- **GAUSS: GrAph-customized Universal Self-Supervised Learning** (WWW 2024)
- Authors: Liang Yang et al.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `hidden_dim` or `max_neighbors` in ego-network construction
2. **Slow training**: Reduce `max_iter` or use smaller datasets for testing
3. **Poor performance**: Try different `num_blocks` values based on dataset homophily
4. **Installation issues**: Make sure PyTorch Geometric is properly installed with correct CUDA version

### Performance Tips

- Use GPU for faster training (automatically detected)
- Adjust `num_blocks` based on dataset characteristics
- Use early stopping to prevent overfitting
- For large graphs, consider sampling strategies (not implemented in this version)

## 📚 References

1. Yang, L., et al. (2024). GAUSS: GrAph-customized Universal Self-Supervised Learning. WWW 2024.
2. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017.
3. Zhu, Y., et al. (2020). Deep graph contrastive representation learning. arXiv preprint.

---

**Note**: This is a research implementation. For production use, consider additional optimizations and error handling.
