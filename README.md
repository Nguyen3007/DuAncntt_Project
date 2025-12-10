# Recommendation System with LightGCN and NGCF

A PyTorch implementation of graph-based collaborative filtering models for recommendation systems, featuring LightGCN and NGCF (Neural Graph Collaborative Filtering) algorithms.

## Overview

This project implements state-of-the-art graph neural network models for collaborative filtering:

- **LightGCN**: A simplified and effective Graph Convolutional Network for recommendation
- **NGCF**: Neural Graph Collaborative Filtering with explicit modeling of high-order connectivity
- **ALS**: Alternative Least Squares as a baseline method

The models are designed for implicit feedback scenarios and use Bayesian Personalized Ranking (BPR) loss for optimization.

## Features

- **Multiple Model Implementations**
  - LightGCN: Lightweight graph convolution for collaborative filtering
  - NGCF: Neural graph collaborative filtering with message passing
  - ALS: Traditional matrix factorization baseline

- **Advanced Graph Building**
  - Standard graph construction from user-item interactions
  - Time-decay graph builder for temporal dynamics

- **Comprehensive Training Pipeline**
  - BPR (Bayesian Personalized Ranking) loss optimization
  - Support for training resumption
  - Configurable hyperparameters
  - GPU acceleration support

- **Evaluation Metrics**
  - Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - Precision@K

## Project Structure

```
DuAncntt_Project/
├── src/
│   ├── models/
│   │   ├── LightGCN.py          # LightGCN model implementation
│   │   └── NGCF.py              # NGCF model implementation
│   └── data_utils/
│       ├── dataloader.py        # Data loading utilities
│       ├── graph_builder.py     # Standard graph construction
│       └── graph_builder_time_decay.py  # Time-decay graph builder
├── trainer/
│   ├── train_lightGCN_v2.py     # LightGCN training script
│   ├── train_ngcf.py            # NGCF training script
│   ├── train_als.py             # ALS training script
│   └── resume_lightgcn.py       # Resume training from checkpoint
├── evaluate/
│   ├── evaluate_lightgcn.py     # LightGCN evaluation
│   └── evaluate_ngcf.py         # NGCF evaluation
├── data/
│   ├── h_m/                     # H&M dataset
│   └── vibrent/                 # Vibrent dataset
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Nguyen3007/DuAncntt_Project.git
cd DuAncntt_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Format

The data should be in text format with each line representing a user and their interacted items:
```
user_id item1 item2 item3 ...
```

Place your data files in the `data/` directory:
- `train.txt`: Training data
- `test.txt`: Test data (optional)

### Training Models

#### Train LightGCN

```bash
python trainer/train_lightGCN_v2.py \
    --data_dir data/your_dataset \
    --emb_dim 64 \
    --n_layers 3 \
    --lr 0.001 \
    --reg_weight 1e-4 \
    --batch_size 2048 \
    --epochs 1000
```

#### Train NGCF

```bash
python trainer/train_ngcf.py \
    --data_dir data/your_dataset \
    --emb_dim 64 \
    --layer_sizes 64,64 \
    --lr 0.0001 \
    --reg_weight 1e-5 \
    --batch_size 1024 \
    --epochs 400
```

#### Train ALS (Baseline)

```bash
python trainer/train_als.py \
    --data_dir data/your_dataset \
    --factors 64 \
    --regularization 0.01 \
    --iterations 15
```

### Evaluation

#### Evaluate LightGCN

```bash
python evaluate/evaluate_lightgcn.py \
    --data_dir data/your_dataset \
    --checkpoint path/to/checkpoint.pth \
    --top_k 20
```

#### Evaluate NGCF

```bash
python evaluate/evaluate_ngcf.py \
    --data_dir data/your_dataset \
    --checkpoint path/to/checkpoint.pth \
    --top_k 20
```

### Resume Training

To resume training from a checkpoint:

```bash
python trainer/resume_lightgcn.py \
    --data_dir data/your_dataset \
    --checkpoint path/to/checkpoint.pth \
    --epochs 500
```

## Model Details

### LightGCN

LightGCN simplifies the design of graph convolutional networks for collaborative filtering by:
- Removing feature transformation and nonlinear activation
- Using only the normalized sum of neighbor embeddings
- Combining embeddings from all layers as the final representation

Key hyperparameters:
- `emb_dim`: Embedding dimension (default: 64)
- `n_layers`: Number of graph convolutional layers (default: 3)
- `reg_weight`: L2 regularization weight (default: 1e-4)

### NGCF

NGCF explicitly models high-order connectivity in the user-item graph:
- Propagates embeddings through multiple graph convolutional layers
- Incorporates feature transformation and interaction
- Uses message dropout for regularization

Key hyperparameters:
- `emb_dim`: Initial embedding dimension (default: 64)
- `layer_sizes`: Dimensions of each GCN layer (default: [64, 64])
- `mess_dropout`: Message dropout rate (default: 0.1)
- `reg_weight`: L2 regularization weight (default: 1e-5)

## Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled GPU for faster training
2. **Batch Size**: Larger batch sizes generally improve training stability
3. **Learning Rate**: Start with smaller learning rates (0.0001-0.001)
4. **Regularization**: Adjust `reg_weight` to prevent overfitting
5. **Early Stopping**: Monitor validation metrics to prevent overtraining

## Dependencies

- PyTorch: Deep learning framework
- NumPy: Numerical computations
- Pandas: Data manipulation
- SciPy: Sparse matrix operations (for ALS)
- scikit-learn: Additional utilities

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## Citation

If you use this code in your research, please cite the original papers:

**LightGCN:**
```bibtex
@inproceedings{he2020lightgcn,
  title={LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
  author={He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={639--648},
  year={2020}
}
```

**NGCF:**
```bibtex
@inproceedings{wang2019neural,
  title={Neural Graph Collaborative Filtering},
  author={Wang, Xiang and He, Xiangnan and Wang, Meng and Feng, Fuli and Chua, Tat-Seng},
  booktitle={Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={165--174},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LightGCN and NGCF papers and their authors
- PyTorch team for the excellent deep learning framework
- Contributors and users of this repository

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.
