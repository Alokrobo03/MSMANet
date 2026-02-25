"""
Training script for MSMANet

Example usage:
    python train.py --dataset taxibj --epochs 100
    python train.py --dataset bair --epochs 100
"""

import argparse
from model import build_msmanet
import tensorflow as tf

def train_msmanet(dataset='taxibj', epochs=100):
    
    
    configs = {
        'taxibj': {
            'input_shape': (4, 32, 32, 2),
            'output_frames': 4,
            'batch_size': 32
        },
        'kth': {
            'input_shape': (10, 128, 128, 1),
            'output_frames': 10,
            'batch_size': 16
        },
        'bair': {
            'input_shape': (2, 64, 64, 3),
            'output_frames': 14,
            'batch_size': 32
        },
        'mnist': {
            'input_shape': (10, 64, 64, 1),
            'output_frames': 10,
            'batch_size': 64
        }
    }
    
    config = configs[dataset]
    
    print(f"Building MSMANet for {dataset}...")
    model = build_msmanet(
        input_shape=config['input_shape'],
        output_frames=config['output_frames'],
        filters=[128, 128, 128, 64]
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    # Load dataset here
    # train_data = load_data(dataset, split='train')
    # val_data = load_data(dataset, split='val')
    
    
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MSMANet')
    parser.add_argument('--dataset', type=str, default='taxibj',
                        choices=['taxibj', 'kth', 'bair', 'mnist'],
                        help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    
    args = parser.parse_args()
    
    model = train_msmanet(args.dataset, args.epochs)
