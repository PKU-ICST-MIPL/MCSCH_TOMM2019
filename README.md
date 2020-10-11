# Introduction
This is the source code of our CVPR 2019 paper "Sequential Cross-Modal Hashing Learning via Multi-scale Correlation Mining". Please cite the following paper if you use our code.

Zhaoda Ye and Yuxin Peng, "Sequential Cross-Modal Hashing Learning via Multi-scale Correlation Mining", ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), Vol. 15, No. 4, p. 105, Dec. 2019.(SCI, EI)

# Dependency

This code is implemented with pytorch.

# Data Preparation

The codes adpots the extracted features as input, the details of the features can be found in the paper.

The feature file is a matrix, each line contains a single feature of the i-th data.

# Usage

Start training and tesing by executiving the following commands. This will train and test the model on MIRFlickr datatset. 

python train.py 3 32 32 mir

train.py [gpu_id,bit_length,batch_size,dataset]


