import numpy as np
from .distance import l2_distance, cosine_distance

class IndexFlat:
    def __init__(self, dim, metric='l2'):
        """
        初始化 Flat 暴力检索索引
        :param dim: 向量的维度
        :param metric: 距离度量方式，'l2' 或 'cosine'
        """
        self.dim = dim
        self.metric = metric
        
        # 💡 工业级数据库设计：
        # 我们将内存分为 "不可变基础段 (base)" 和 "可变新增段 (vectors)"
        # mmap 加载的文件是只读的，不能直接往里追加，所以新写入的数据存在这里。
        self._base_matrix = None # 用于承载从磁盘 mmap 加载的数据 (不可变)
        self.vectors = []        # 存放内存中新添加的向量 (可变)
        
        self.payloads = []       # 所有的 payload 元数据
        self._need_rebuild = False
        self._vector_matrix = None # 最终用于搜索的拼接后矩阵
        
    @property
    def total_count(self):
        """获取总向量条数"""
        base_len = self._base_matrix.shape[0] if self._base_matrix is not None else 0
        return base_len + len(self.vectors)
        
    def add(self, vector, payload=None):
        """添加单个向量"""
        assert len(vector) == self.dim, f"向量维度必须为 {self.dim}"
        self.vectors.append(np.array(vector, dtype=np.float32))
        self.payloads.append(payload)
        self._need_rebuild = True
        return self.total_count - 1
        
    def add_batch(self, vectors, payloads=None):
        """批量添加向量"""
        vectors = np.array(vectors, dtype=np.float32)
        assert vectors.shape[1] == self.dim
        self.vectors.extend(vectors)
        if payloads:
            assert len(payloads) == len(vectors)
            self.payloads.extend(payloads)
        else:
            self.payloads.extend([None] * len(vectors))
        self._need_rebuild = True

    def _build_matrix(self):
        """将不可变的基础段(mmap)与内存新增段拼接成一个统一的搜索矩阵"""
        if self._need_rebuild:
            new_matrix = np.vstack(self.vectors) if len(self.vectors) > 0 else None
            
            if self._base_matrix is not None and new_matrix is not None:
                # 拼接：上方是 mmap 映射的旧数据，下方是新数据
                self._vector_matrix = np.vstack([self._base_matrix, new_matrix])
            elif self._base_matrix is not None:
                self._vector_matrix = self._base_matrix
            elif new_matrix is not None:
                self._vector_matrix = new_matrix
            else:
                self._vector_matrix = np.empty((0, self.dim), dtype=np.float32)
                
            self._need_rebuild = False
        elif self._vector_matrix is None and self._base_matrix is not None:
            self._vector_matrix = self._base_matrix

    def search(self, query, top_k=5):
        """搜索最相似的 top_k 个向量"""
        if self.total_count == 0:
            return [], []
            
        self._build_matrix()
        query = np.array(query, dtype=np.float32)
        
        # 1. 计算距离
        if self.metric == 'l2':
            distances = l2_distance(query, self._vector_matrix)
        elif self.metric == 'cosine':
            distances = cosine_distance(query, self._vector_matrix)
        else:
            raise ValueError(f"不支持的度量方式: {self.metric}")
            
        # 2. 找出距离最小的 top_k 个
        k = min(top_k, self.total_count)
        
        if k == self.total_count:
            sorted_indices = np.argsort(distances)
            return distances[sorted_indices], [self.payloads[i] for i in sorted_indices]
            
        # O(N) 复杂度的前 k 个过滤
        top_k_indices_unordered = np.argpartition(distances, k - 1)[:k]
        
        # O(k log k) 局部排序
        top_k_distances_unordered = distances[top_k_indices_unordered]
        local_sorted_indices = np.argsort(top_k_distances_unordered)
        
        # 映射回全局索引
        final_indices = top_k_indices_unordered[local_sorted_indices]
        
        result_distances = distances[final_indices]
        result_payloads = [self.payloads[i] for i in final_indices]
        
        return result_distances, result_payloads
