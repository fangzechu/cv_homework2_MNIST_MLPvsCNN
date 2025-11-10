
# MNIST: MLP vs CNN 对比（PyTorch）
cv课的第二次作业，比较MLP和CNN模型
## 模型
- **MLP_A**: 784-512-256-10（层数与 CNN_A 接近，但参数量大得多）
- **CNN_A**: conv1(1→4,k=5)-pool-conv2(4→8,k=5)-pool-fc（浅层小通道；准确率刻意不冲高）
- **MLP_B**: 784-8-10（参数量与 CNN_B 接近）
- **CNN_B**: conv1(1→8,k=5)-pool-conv2(8→16,k=5)-pool-fc（参数量与 MLP_B 同量级）

## 期望现象
1. **层数差不多时**：`MLP_A` 参数量 ≫ `CNN_A`；精度上 `MLP_A` ≳ `CNN_A`（差距很小，常见为 0~1%）。
2. **参数量差不多时**：`CNN_B` ≫ `MLP_B`（一般高出数个百分点）。

> 注意：精度具体数值与训练轮次、数据增强等有关。若想抑制 CNN_A 精度，请：
> - 使用本文提供的极小通道数（4、8），
> - 训练轮次 3~5，
> - 不使用 BatchNorm/Dropout/强增强。

## 运行
```bash
python train_mnist.py --arch MLP_A  --epochs 5
python train_mnist.py --arch CNN_A  --epochs 5
python train_mnist.py --arch MLP_B  --epochs 5
python train_mnist.py --arch CNN_B  --epochs 5
```

## 每层输入/输出尺寸与参数量
已在 `mnist_model_layers.csv` 给出（由前向 hooks 自动统计）。
