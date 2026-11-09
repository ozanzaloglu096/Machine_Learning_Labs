# Lab 3: Model Architectures, Performance, and Trade-off Analysis Report

## 1. Introduction
This lab compares three different deep learning architectures (**MLP**, **CNN**, and a lightweight **Attention/Transformer-based model**) using the Fashion-MNIST dataset. The evaluation focuses on balancing accuracy, parameter counts, FLOPs, and inference latency.

## 2. Model Architectures & Complexity
The complexity metrics measured via `torchinfo` and `thop` are summarized below:

| Model | Parameters | FLOPs (MACs $\times 2$) | Latency (Batch 1, Median) | Latency (Batch 128, Median) |
| :--- | :--- | :--- | :--- | :--- |
| **MLP** | 235,146 | 469,504.0 | 0.03 ms | 0.09 ms |
| **CNN** | 20,490 | 2,063,488.0 | 0.11 ms | 4.08 ms |
| **Attention** | 78,794 | 1,182,976.0 | 0.30 ms | 4.48 ms |

## 3. Training & Test Accuracy Results
After training/evaluating the models on the Fashion-MNIST dataset, the test accuracies achieved are:
* **MLP Test Accuracy:** ~85.0%
* **CNN Test Accuracy:** ~92.3%
* **Attention Test Accuracy:** ~89.1%

## 4. Trade-off Analysis (Parameters & Latency vs. Accuracy)
To visualize the trade-offs, `tradeoff_plots.png` was generated, comparing parameter counts and batch execution latencies against test accuracies.

![Trade-off Plots](tradeoff_plots.png)

* **Parameters vs. Accuracy:** The CNN achieves the highest accuracy with the lowest parameter footprint, proving highly efficient for spatial image data. The MLP requires significantly more parameters due to fully connected layers while yielding lower accuracy.
* **Latency vs. Accuracy:** While MLP offers the fastest inference due to simple matrix multiplications, CNN and Attention models introduce computational overhead while providing superior feature extraction capabilities.

## 5. Conclusion
The convolutional architecture (CNN) provides the best overall trade-off between accuracy and hardware efficiency for image classification tasks on Fashion-MNIST.
