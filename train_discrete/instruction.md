# Buffer vs JPGDataset 数据对比

## 1. 输出数据对比

### Buffer 原始 .pt 文件 (DataCollection.py 产出 / export_legacy_buffer_pt 产出)

**由 `torch.load()` 读入后直接得到**，每个 `.pt` 文件对应一个完整 episode，是 `load_Buffer()` 的实际输入：

```python
{
    "image":  list of np.ndarray (H, W, 3),    # RGB/BGR 图像序列 (uint8)
    "action": list of np.ndarray (1, 2),        # 连续动作 (vx, vy)  (float32/float64)
    "reward": list of np.ndarray (1,),          # 标量奖励             (float32/float64)
    "is_first": np.ndarray (T,) bool,           # 首帧标记 (首帧=True, 其余=False)
    "is_last":  np.ndarray (T,) bool,           # 末帧标记 (末帧=True, 其余=False)
}
```

| 键 | 每元素类型/Shape | dtype | 含义 |
|----|-----------------|-------|------|
| image | `(H, W, 3)` uint8 | uint8 | BGR 图像，H/W 可任意（如 240×240） |
| action | `(1, 2)` float32 | float32 / float64 | 连续动作 (vx, vy)，范围约 [-30,30] / [-100,100] |
| reward | `(1,)` float32 | float32 / float64 | 标量奖励 |
| is_first | `(T,)` bool | bool | 首帧为 True，其余为 False |
| is_last | `(T,)` bool | bool | 末帧为 True，其余为 False |

**参数说明**：
- T: episode 长度（每文件固定，如 500）
- H, W: 可变，取决于采集时的分辨率（如 240×240）
- 图像格式：`(H, W, C)`，BGR 或 RGB

---

### `load_Buffer()` 处理过程

`train_offline.py` 的 `load_Buffer()` 读取上述 `.pt` 文件后，逐条构造 experience 存入 `ReplayBuffer`：

```
dict_tmp = torch.load(pt_path)
                              image[:-1] → resize(64,64) → transpose(2,0,1) → state     [3,64,64]
                              image[1:]  → resize(64,64) → transpose(2,0,1) → next_state [3,64,64]
dict_tmp  ──→  action ──→ np.array(action)[:-1].squeeze(1)                   → act       [2]
               reward ──→ np.array(reward).squeeze(1)[:-1]                    → re        [1]
               (done 由循环内逻辑生成，当前实现始终为 False)
```

`load_Buffer()` 处理完成后 → `ReplayBuffer.add(state, act, re, next_state, done)` → 存入 deque。

**`ReplayBuffer.sample()` 输出**（训练时实际使用的 batch）：
```python
(states, actions, rewards, next_states, dones)
```

| Tensor | Shape | 含义 |
|--------|-------|------|
| states | `[batch_size, lstm_seq_len, 3, 64, 64]` | 图像序列 |
| actions | `[batch_size, lstm_seq_len, 2]` | 连续动作 (vx, vy) |
| rewards | `[batch_size, lstm_seq_len, 1]` | 奖励值 |
| next_states | `[batch_size, lstm_seq_len, 3, 64, 64]` | 下一帧图像 |
| dones | `[batch_size, lstm_seq_len, 1]` | 终止标志 |

**参数说明**：
- batch_size=32, lstm_seq_len=20
- 图像：64×64 RGB，格式 (C, H, W)
- 动作：2 维连续值，范围 [-30,30] 和 [-100,100]

---

### JPGDataset (当前版本)

**采样输出** `dataset[idx]` 返回：
```python
(states, actions, relative_pos, dones)
```

| Tensor | Shape | 含义 |
|--------|-------|------|
| states | `[T, 512, 512, 3]` | RGB 图像序列 |
| actions | `[T]` | 专家动作序列 (0-8) |
| relative_pos | `[T, 3]` | 相对位置 (x, y, z) |
| dones | `[T]` | 终止标记 (最后一个位置为 1，其余为 0) |

**参数说明**：
- T: episode 长度（不固定，可 padding）
- 图像：512x512，格式 (H, W, C)
- 动作：离散值 0-8
- `dones`: 序列内最后一个时间步为 1（表示 episode 结束），其他为 0

---

### JPGDataset → Buffer 转换桥 (export_legacy_buffer_pt)

**导出方法** `dataset.export_legacy_buffer_pt()` 将 JPGDataset 数据转换为 `.pt` 文件，格式可直接被 `train_offline.py` 的 `load_Buffer()` 加载。

**输出 `.pt` 文件结构**：
```python
{
    "image":    list of np.ndarray (export_img_size, export_img_size, 3),  # BGR 图像 (uint8)
    "action":   list of np.ndarray (1, 2),       # 连续动作 (vx, vy)  (float32)
    "reward":   list of np.ndarray (1,),         # 标量奖励             (float32)
    "is_first": np.ndarray (T,) bool,            # 首帧标记
    "is_last":  np.ndarray (T,) bool,            # 末帧标记
}
```

**`load_Buffer()` 解析逻辑**：

| 数据 | 导出格式 (每元素) | `load_Buffer()` 处理 | 最终 Shape |
|------|------------------|---------------------|------------|
| image | `(export_img_size, export_img_size, 3)` uint8 | `[:-1]` → resize 64×64 → transpose(C,H,W) | `(T-1, 3, 64, 64)` |
| action | `(1, 2)` float32 | `[:-1]` → `np.array` → `squeeze(axis=1)` | `(T-1, 2)` |
| reward | `(1,)` float32 | `squeeze(axis=1)[:-1]` | `(T-1,)` |

**参数说明**：
- `export_img_size`: 默认 240，可通过参数控制
- `to_bgr`: 默认 True，RGB → BGR（匹配 `DataCollection.py` 采集格式）
- 使用 torchvision (`TF.resize`) 进行图像缩放，无需 cv2

**转换细节**：
1. **动作映射**：离散动作 0-8 → 连续动作 (vx, vy)，映射表见 `ACTION_TO_LEGACY_CONTINUOUS`
2. **奖励计算**：基于 `relative_pos` 计算 tracking reward（与目标和期望距离/角度相关）
3. **图像缩放**：在 text processor 处理后、append 之前，用 `TF.resize` 缩放到 `export_img_size × export_img_size`
4. **色彩空间**：RGB → BGR（通过 `[:, :, ::-1]`），匹配 `DataCollection.py` 的 OpenCV BGR 格式
5. **`next_state`**：通过 `image[1:]` 隐式构造，`state = image[:-1]`, `next_state = image[1:]`
6. **`done` 标记**：`load_Buffer()` 内部在循环中逐条添加，但当前逻辑始终为 False
7. **`is_first` / `is_last`**：导出时自动标记，格式为 `ndarray (T,) bool`，`load_Buffer()` 不使用这两个字段

---

### export_legacy_buffer_pt 输出 vs 参考 .pt 文件格式对齐表

以下与 `/data/qingcheng.zhu/Offline_RL_Active_Tracking/inperfect_expert_240px_v4_deva_mask_v1/v1_inperfect_expert_64px_v4_0002-500.pt` 逐项对比：

| 字段 | 参考文件 | export_legacy_buffer_pt | 对齐？ |
|------|---------|------------------------|--------|
| **keys** | `action`, `image`, `reward`, `is_first`, `is_last` | 同上 5 keys | ✅ |
| **image 每元素 shape** | `(240, 240, 3)` | `(export_img_size=240, ...)` | ✅ |
| **image 每元素 dtype** | `uint8` | `uint8` | ✅ |
| **image 色彩** | BGR (OpenCV 采集) | BGR (`to_bgr=True` RGB→BGR) | ✅ |
| **action 每元素 shape** | `(1, 2)` | `(1, 2)` | ✅ |
| **action 每元素 dtype** | `float64` | `float32` | ⚠️ 精度不同，不影响 `load_Buffer()` |
| **reward 每元素 shape** | `(1,)` | `(1,)` | ✅ |
| **reward 每元素 dtype** | `float64` | `float32` | ⚠️ 精度不同，不影响 `load_Buffer()` |
| **is_first** | `ndarray (T,)` bool | `ndarray (T,)` bool | ✅ |
| **is_last** | `ndarray (T,)` bool | `ndarray (T,)` bool | ✅ |

> ⚠️ 参考文件的 action/reward 为 `float64`，当前导出为 `float32`。`load_Buffer()` 内部通过 `torch.from_numpy(...).float()` 转为 `float32`，因此不影响训练。

---

## 2. 核心差异（原始 .pt 文件 vs JPGDataset）

| 维度 | 原始 .pt 文件 (load_Buffer 输入) | JPGDataset |
|------|----------------------------------|------------|
| **文件格式** | dictionary (6 keys) | `__getitem__()` 返回 tuple |
| **图像尺寸** | 可变 (如 240×240) | 512×512 |
| **图像格式** | (H,W,C) uint8 | (H,W,C) uint8 |
| **动作类型** | 连续 (2 维) | 离散 (0-8) |
| **奖励** | ✓ 有 | ✗ 无 |
| **done** | load_Buffer 内部生成 | ✓ 有 (最后一个为 1) |
| **next_state** | image[1:] 隐式构造 | ✗ 无 |
| **relative_pos** | ✗ 无 | ✓ 有 |
| **额外键** | is_first, is_last | 无 |

**原始 .pt 文件两来源对比**：

| 来源 | action 形状 | image 尺寸 | 额外字段 |
|------|------------|-----------|---------|
| `DataCollection.py` 在线采集 | `(1, 2)` float32/float64 | 如 240×240 (BGR) | `is_first`, `is_last` (bool) |
| `export_legacy_buffer_pt()` 导出 | `(1, 2)` float32 | `export_img_size` 可控 (默认 240×240, BGR) | `is_first`, `is_last` (bool) |

两者现已完全对齐，均含 `action`/`image`/`reward`/`is_first`/`is_last` 五个键，均可被 `load_Buffer()` 读取。

---

## 3. 结论

**`JPGDataset.__getitem__()` 与原始 .pt 文件格式差异巨大**：
- 原始 .pt 为 **离线 RL** 设计的 (image, action_cont, reward, is_first, is_last) 五元组
- JPGDataset 为 **模仿学习** 设计：(obs, expert_action_discrete, relative_pos)

**`export_legacy_buffer_pt()` 已作为转换桥完成格式对齐**：
- 输出的 `.pt` 文件包含 `image`/`action`/`reward`/`is_first`/`is_last` 五个键，与 `DataCollection.py` 产出的参考文件完全一致
- `image` 缩放到 240×240（可配置），BGR 格式，`uint8`
- `action` 为 `(1, 2)` 连续向量，已自动完成离散→连续映射
- `reward` 为 `(1,)` 标量，基于 `relative_pos` 计算 tracking reward
- `is_first`/`is_last` 为 `ndarray (T,) bool`
- 可直接 `export_legacy_buffer_pt()` → `load_Buffer()` 链路训练

**如需直接使用 `JPGDataset.__getitem__()` 训练**，仍需要：
1. 构造连续动作（需离散→连续映射）
2. 构造 reward（原始数据无奖励）
3. 构造 next_state（原始数据无）
4. 图像缩放 + 转置 (H,W,C)→(C,H,W)
