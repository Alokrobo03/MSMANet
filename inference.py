

import numpy as np
from model import build_msmanet
import tensorflow as tf


def predict_taxibj_example():
   
    
    model = build_msmanet(
        input_shape=(4, 32, 32, 2),
        output_frames=4
    )
    
    predictions = model.predict(context)
    
    print(f"Input shape: {context.shape}")
    print(f"Output shape: {predictions.shape}")
    print("✓ Prediction successful!")
    
    return predictions


def predict_bair_example():
   
    model = build_msmanet(
        input_shape=(2, 64, 64, 3),
        output_frames=14
    )
    
    predictions = model.predict(context)
    
    print(f"Input shape: {context.shape}")
    print(f"Output shape: {predictions.shape}")
    print("✓ Prediction successful!")
    
    return predictions


if __name__ == '__main__':
    
    predict_taxibj_example()
    
    predict_bair_example()