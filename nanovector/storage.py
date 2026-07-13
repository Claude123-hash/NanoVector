import os
import struct
import pickle
import numpy as np

MAGIC_HEADER = b'NANOVEC1'
HEADER_SIZE = 128

def save_index_flat(index, directory):
    """
    将 IndexFlat 持久化到硬盘。
    我们使用两个文件：
    1. vectors.bin: 存储 Header 和紧凑的 NumPy 浮点数矩阵 (专为 mmap 设计)
    2. payloads.pkl: 存储元数据 (Python 字典)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    index._build_matrix() # 确保矩阵已完全构建
    
    vec_path = os.path.join(directory, 'vectors.bin')
    payload_path = os.path.join(directory, 'payloads.pkl')
    
    # 1. 保存 payloads (使用简单的 pickle)
    with open(payload_path, 'wb') as f:
        pickle.dump(index.payloads, f)
        
    # 2. 保存向量矩阵 (二进制格式)
    with open(vec_path, 'wb') as f:
        matrix = index._vector_matrix
        N = matrix.shape[0] if matrix is not None else 0
        D = index.dim
        metric_bytes = index.metric.ljust(4).encode('ascii')[:4]
        
        # 打包 Header:
        # <8sII4s = 小端序, 8字节字符串(magic), 无符号整型(N), 无符号整型(D), 4字节字符串(metric)
        header_pack = struct.pack('<8sII4s', MAGIC_HEADER, N, D, metric_bytes)
        # 用 0 填充，对齐到 128 字节，保证后面的矩阵数据按页对齐，有利于 mmap 性能
        header_pack = header_pack.ljust(HEADER_SIZE, b'\x00')
        f.write(header_pack)
        
        # 极速写入：直接把 NumPy 内存块(tobytes) dump 到磁盘
        if N > 0:
            f.write(matrix.tobytes())

def load_index_flat(directory, use_mmap=True):
    """
    从硬盘加载 IndexFlat。
    如果 use_mmap=True，将开启大厂常用的“零拷贝技术(内存映射)”，
    启动速度极快，不占用大量 RAM 内存。
    """
    from .index_flat import IndexFlat
    
    vec_path = os.path.join(directory, 'vectors.bin')
    payload_path = os.path.join(directory, 'payloads.pkl')
    
    if not os.path.exists(vec_path) or not os.path.exists(payload_path):
        raise FileNotFoundError("找不到 NanoVector 索引文件")
        
    # 1. 读取头信息
    with open(vec_path, 'rb') as f:
        header_bytes = f.read(HEADER_SIZE)
        magic, N, D, metric = struct.unpack('<8sII4s', header_bytes[:20])
        
        if magic != MAGIC_HEADER:
            raise ValueError("不合法的 NanoVector 索引文件格式")
            
        metric = metric.decode('ascii').strip()
        
    index = IndexFlat(dim=D, metric=metric)
    
    # 2. 恢复 payloads
    with open(payload_path, 'rb') as f:
        index.payloads = pickle.load(f)
        
    # 3. 恢复向量矩阵
    if N > 0:
        if use_mmap:
            # 💡 核心黑科技：np.memmap 
            # 此时并不会把整个上 G 的文件读进内存，而是按需映射 (Page Fault)，
            # 这是所有现代数据库冷启动秒开的底层秘密。
            index._base_matrix = np.memmap(vec_path, dtype='float32', mode='r', offset=HEADER_SIZE, shape=(N, D))
        else:
            # 常规的全量加载 (用于对比)
            index._base_matrix = np.fromfile(vec_path, dtype='float32', offset=HEADER_SIZE).reshape(N, D)
            
        index._vector_matrix = index._base_matrix
        
    return index
