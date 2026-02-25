# Datasets Used 

This document provides details for all datasets used in our experiments.

## 1. TaxiBJ (Traffic Flow Prediction)

**Source:** Beijing taxi GPS trajectories (2013-2016)

**Official Download:** [IEEE DataPort](https://ieee-dataport.org/documents/traffic-datsets-abilene-geant-taxibj)

**Format:**
- Input: 4 frames × 32×32 × 2 channels (inflow/outflow)
- Output: 4 future frames
- Total: ~22,000 samples
- Time interval: 30 minutes per frame
- Spatial resolution: 32×32 grid covering Beijing





## 2. KTH Actions (Human Motion Videos)

**Source:** KTH Royal Institute of Technology

**Download:** https://www.csc.kth.se/cvap/actions/

**Direct link:** https://www.csc.kth.se/cvap/actions/walking.zip (repeat for other actions)

**Actions:** walking, jogging, running, boxing, hand waving, hand clapping

**Format:**
- Input: 10 frames × 128×128 × 1 channel (grayscale)
- Output: 10 future frames
- Videos: 599 sequences, 25 subjects



---

## 3. BAIR Robot Pushing

**Source:** UC Berkeley AI Research Lab

**Download:**
```bash
wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
tar -xvf bair_robot_pushing_dataset_v0.tar
```


**Format:**
- Input: 2 frames × 64×64 × 3 channels (RGB)
- Output: 14 future frames
- Total: 44,000 training + 256 test sequences


---

## 4. Moving MNIST (Digit Sequences)

**Source:** Official Moving MNIST test set

**Download:** 
```bash
# Official test set (10,000 sequences)
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
```


**Official Repository:** [Toronto - Unsupervised Video](http://www.cs.toronto.edu/~nitish/unsupervised_video/)

**Format:**
- Pre-generated sequences: 10,000 test samples
- Input: 10 frames × 64×64 × 1 channel (grayscale)
- Output: 10 future frames
- Total frames per sequence: 20


