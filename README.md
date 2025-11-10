
# MNIST: MLP vs CNN 对比（PyTorch）
cv课的第二次作业，比较MLP和CNN模型
## 模型
- **MLP_A**: 784-512-256-10（层数与 CNN_A 接近，但参数量大得多）
- **CNN_A**: conv1(1→4,k=5)-pool-conv2(4→8,k=5)-pool-fc（层数与 MLP_A 接近，但参数量小得多）
- **MLP_B**: 784-8-10（参数量与 CNN_B 接近）
- **CNN_B**: conv1(1→8,k=5)-pool-conv2(8→16,k=5)-pool-fc（参数量与 MLP_B 同量级）

## 实验结果分析
1. **层数差不多时**：`MLP_A` 参数量 ≫ `CNN_A`；精度上 `MLP_A`与 `CNN_A`差距很小。
2. **参数量差不多时**：精度上`CNN_B` ≫ `MLP_B`。

。

## 运行
```bash
python train_mnist.py --arch MLP_A  --epochs 5
python train_mnist.py --arch CNN_A  --epochs 5
python train_mnist.py --arch MLP_B  --epochs 5
python train_mnist.py --arch CNN_B  --epochs 5
```

## 每层输入/输出尺寸，参数量与准确率
已在 [`mnist_model_layers.csv`](mnist_model_layers.csv) 给出（由前向 hooks 自动统计）。
