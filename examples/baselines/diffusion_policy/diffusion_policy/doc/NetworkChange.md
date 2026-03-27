
改动 |	需要重新生成  |  Demo |	预期提升|	复杂度
✅ 已改：PlainConv AvgPool	否	中	低
预训练 ResNet18	否	高	中
unet_dims 增大	否	中	低
visual_feature_dim: 256→512	否	中	低
ColorJitter 数据增强	否	中（针对干扰物）	低
建议的改动顺序：先把 unet_dims 和 visual_feature_dim 改大（改两个参数就行），再考虑换 ResNet18 encoder。

```python
python examples/baselines/diffusion_policy/train_stackcube.py     --demo-path videos/StackCube-v1/stackcube_expert.rgb.pd_ee_delta_pos.physx_cpu.h5     --num-eval-episodes 20     --total-iters 500000     --num-demos 100 --unet-dims 128 256 512
```

`visual_feature_dim`
视觉编码器（PlainConv）的输出向量维度，也就是把一帧 128×128 的图像压缩成多少维的特征。
这个向量会和 state（qpos/qvel/tcp_pose）拼接，作为 UNet 的全局条件（FiLM conditioning）

`unet_dims`
UNet 每层的通道数，决定了噪声预测网络的容量。
默认是 `[64, 128, 256]`，总参数量约 4.5M。改成 `[128, 256, 512]` 会翻倍，达到 9M 左右，通常能提升性能。

