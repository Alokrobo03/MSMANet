"""
Testing/Inference script for MSMANet

Example usage:
    python test.py --dataset taxibj --weights weights/taxibj_msmanet.h5
    python test.py --dataset bair --weights weights/bair_msmanet.h5
"""

import argparse
import numpy as np
from model import build_msmanet, ChannelAttention, \
                  MultiScaleMotionMagnitudeModule, MultiScaleMotionDirectionModule
import tensorflow as tf


def test_msmanet(dataset='taxibj', weights_path=None):
   
    configs = {
        'taxibj': {'input_shape': (4, 32, 32, 2), 'output_frames': 4},
        'kth': {'input_shape': (10, 128, 128, 1), 'output_frames': 10},
        'bair': {'input_shape': (2, 64, 64, 3), 'output_frames': 14},
        'mnist': {'input_shape': (10, 64, 64, 1), 'output_frames': 10}
    }
    
    config = configs[dataset]
    
    
    print(f"Building MSMANet for {dataset}...")
    model = build_msmanet(
        input_shape=config['input_shape'],
        output_frames=config['output_frames']
    )
    
   
    if weights_path:
        print(f"Loading weights from {weights_path}...")
        
        custom_objects = {
            'ChannelAttention': ChannelAttention,
            'MultiScaleMotionMagnitudeModule': MultiScaleMotionMagnitudeModule,
            'MultiScaleMotionDirectionModule': MultiScaleMotionDirectionModule
        }
        
        try:
            model = tf.keras.models.load_model(weights_path, custom_objects=custom_objects)
            print("Weights loaded successfully")
        except:
            model.load_weights(weights_path)
            print(" Weights loaded successfully")
    
    print(f"\nTo complete evaluation:")
    print(f"1. Load {dataset} test dataset")
    print(f"2. Run: predictions = model.predict(test_data)")
    print(f"3. Compute metrics (MSE, PSNR, SSIM, FVD)")
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MSMANet')
    parser.add_argument('--dataset', type=str, default='taxibj',
                        choices=['taxibj', 'kth', 'bair', 'mnist'],
                        help='Dataset to test on')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights')
    
    args = parser.parse_args()
    
    model = test_msmanet(args.dataset, args.weights)
    
    print("\n" + "="*60)
    print("Model ready for inference!")
    print("="*60)
