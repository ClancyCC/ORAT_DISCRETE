# Buffer 提供给 CQLSAC_CNN_LSTM 的数据

## Buffer 采样返回的数据

buffer.sample() 返回一个元组 (states, actions, rewards, next_states, dones)，直接传给 agent.learn()：

```python
experiences = buffer.sample()
agent.learn(experiences)
```

## 数据格式详解

### 1. states
- **类型**: torch.Tensor
- **形状**: [batch_size, lstm_seq_len, 3, 64, 64]
- **内容**: 连续 lstm_seq_len 帧的图像，每帧是 64x64 的 RGB 三通道图像
- **用途**: 经过 CNN_LSTM 编码后提取特征，用于策略和价值网络

### 2. actions
- **类型**: torch.Tensor
- **形状**: [batch_size, lstm_seq_len, 2]
- **内容**: 每个时间步的动作，2 维连续动作（已归一化到 [-1, 1]）
- **用途**: 计算 Q 值

### 3. rewards
- **类型**: torch.Tensor  
- **形状**: [batch_size, lstm_seq_len, 1]
- **内容**: 每个时间步的奖励值
- **用途**: 计算 Q 目标值

### 4. next_states
- **类型**: torch.Tensor
- **形状**: [batch_size, lstm_seq_len, 3, 64, 64]
- **内容**: 下一个时间步的连续图像序列
- **用途**: 计算目标 Q 值

### 5. dones
- **类型**: torch.Tensor
- **形状**: [batch_size, lstm_seq_len, 1]
- **内容**: 每个时间步的终止标志（0 或 1）
- **用途**: 计算 Q 目标值时决定是否截断

## 数据流向

```
buffer.sample()
  ↓
(states, actions, rewards, next_states, dones)
  ↓
agent.learn(experiences)
  ↓
states → CNN_LSTM → 特征 → Actor/Critic
actions → Critic
rewards → Q 目标计算
next_states → CNN_LSTM → 目标特征
dones → Q 目标截断
```
