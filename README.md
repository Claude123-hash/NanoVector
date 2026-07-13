# 🚀 NanoVector (轻量级边缘端向量数据库)

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

**NanoVector** 是一个专为边缘计算和系统研究设计的极简、高性能纯 Python 向量数据库。它去除了沉重的底层 C++ 编译依赖，完全依靠 Python 与 NumPy 的底层 C 语言矩阵计算优势，以及操作系统级的内存映射（mmap）零拷贝技术，实现了海量高维向量的极速近似最近邻（ANN）检索。

## ✨ 核心技术特性 (Key Features)

- **纯 Python & 零依赖**：核心计算仅依赖 `NumPy`，极具跨平台可移植性，是学习底层数据库设计与 ANN 算法的绝佳工程范例。
- **混合段存储架构 (LSM-Tree Inspired)**：借鉴现代顶级数据库引擎架构，采用“不可变基础段 (Base Mmap Segment) + 可变增量段 (Mutable Vectors Segment)”的分离设计，实现无锁的高效数据合并。
- **零拷贝极速冷启动 (Zero-Copy mmap)**：通过 `np.memmap` 实现基于 OS Page Cache 的秒级冷启动，支持在内存受限（如 8GB RAM）的边缘设备上处理远超物理内存容量的超大向量数据文件。
- **纯手写的 IVF 倒排索引与 K-Means++**：抛弃现成的 sklearn 算法库，使用 NumPy 广播机制纯手写了零 for 循环的 K-Means 聚类训练。配合 IVF 倒排文件分桶，让百万级向量的检索速度呈指数级攀升。
- **极致的算法提速细节**：在 Top-K 过滤与排序链条中，使用时间复杂度仅为 $O(N)$ 的 `np.argpartition` 完美替代 $O(N \log N)$ 的传统排序，榨干 CPU 的最后一丝单核算力。

---

## ⚙️ 性能基准测试 (Benchmark)

在普通个人电脑环境下，对 **50,000 条 64 维** 向量进行检索的极限压测报告（执行 `examples/benchmark.py`）：

| 索引算法 | 召回探查 (nprobe) | QPS (每秒查询次数) | 召回率 (Recall) | 较暴力检索提速 |
| :--- | :---: | :---: | :---: | :---: |
| **Flat Index (全局暴力搜索基准)** | - | 154 | 100% | 1.0x |
| **IVF Index (倒排分桶算法)** | **20** | **1683** | **97.8%** | **11倍** |
| **IVF Index (倒排分桶算法)** | 5 | 4293 | 74.6% | 28倍 |
| **IVF Index (倒排分桶算法)** | 1 | 7951 | 27.3% | 51倍 |

> **结论**：在保证 **97.8%** 极高准确召回率的前提下，IVF 引擎相比纯暴力检索提升了整整 **11倍** 速度。展现了算法领域“以微小精度换取巨大空间与时间”的极客艺术。

---

## 🛠️ 快速开始 (Quick Start)

### 安装
```bash
git clone https://github.com/YourUsername/NanoVector.git
cd NanoVector
pip install -r requirements.txt
```

### 极简使用示例

```python
from nanovector import IndexFlat, IndexIVF, save_index_flat, load_index_flat

# 1. 初始化索引 (维度 128, L2 距离度量)
index = IndexFlat(dim=128, metric='l2')

# 2. 插入向量与负载数据 (Payload)
index.add(vector=[0.1, 0.2, ...], payload={"title": "文章 A", "id": 1})
index.add(vector=[0.9, 0.8, ...], payload={"title": "文章 B", "id": 2})

# 3. 极速检索 Top-K
query_vector = [0.1, 0.1, ...]
distances, payloads = index.search(query_vector, top_k=5)

# 4. 工业级持久化落盘
save_index_flat(index, "my_database_dir")

# 5. 秒级零拷贝加载 (mmap)
loaded_index = load_index_flat("my_database_dir", use_mmap=True)
```

## 📂 项目结构
```text
NanoVector_OpenSource/
├── nanovector/          # 核心代码包 (数据库引擎)
│   ├── distance.py      # L2/Cosine 距离数学引擎
│   ├── index_flat.py    # 全局搜索与混合存储核心
│   ├── index_ivf.py     # 倒排文件高级索引
│   ├── kmeans.py        # 纯 NumPy 手写聚类算法
│   └── storage.py       # 二进制序列化与 Mmap 管理
├── examples/            # 压测脚本
│   └── benchmark.py
└── tests/               # 单元测试保证工程质量
```
