import numpy as np
from .index_flat import IndexFlat
from .distance import l2_distance, cosine_distance
from .kmeans import kmeans_train

class IndexIVF:
    def __init__(self, dim, num_clusters, nprobe=1, metric='l2'):
        """
        初始化 IVF 倒排索引
        :param num_clusters: 聚类中心(桶)的数量
        :param nprobe: 搜索时探测最近的几个桶。nprobe 越大越准，但越慢。
        """
        self.dim = dim
        self.num_clusters = num_clusters
        self.nprobe = min(nprobe, num_clusters)
        self.metric = metric
        
        # 聚类中心点矩阵 (K, dim)
        self.centroids = None 
        # 倒排表：每一个聚类（桶）本质上内部就是一个极小的 IndexFlat
        self.buckets = [IndexFlat(dim=dim, metric=metric) for _ in range(num_clusters)]
        
        # 训练前暂存的数据
        self._untrained_vectors = []
        self._untrained_payloads = []
        self.is_trained = False
        
    def add(self, vector, payload=None):
        if not self.is_trained:
            # 还未训练（还没划定聚类中心）时，新数据只能先缓存
            assert len(vector) == self.dim
            self._untrained_vectors.append(np.array(vector, dtype=np.float32))
            self._untrained_payloads.append(payload)
            return -1 # 返回无意义 ID
        else:
            # 已经训练好了，直接找到离自己最近的桶，钻进去！
            vector_np = np.array(vector, dtype=np.float32)
            if self.metric == 'l2':
                dists = l2_distance(vector_np, self.centroids)
            else:
                dists = cosine_distance(vector_np, self.centroids)
                
            closest_bucket_idx = np.argmin(dists)
            self.buckets[closest_bucket_idx].add(vector, payload)
            return closest_bucket_idx

    def train(self):
        """用暂存的数据进行 K-Means 训练并建立倒排表"""
        if self.is_trained:
            return
            
        if len(self._untrained_vectors) < self.num_clusters:
            raise ValueError(f"训练数据量不能少于聚类数！")
            
        X = np.vstack(self._untrained_vectors)
        # 1. 跑 K-Means 算法，训练出最佳的桶位置
        self.centroids = kmeans_train(X, self.num_clusters, metric=self.metric)
        self.is_trained = True
        
        # 2. 将暂存的数据正式分配到对应的桶里
        for i, vec in enumerate(self._untrained_vectors):
            self.add(vec, self._untrained_payloads[i])
            
        # 清空暂存区释放内存
        self._untrained_vectors = []
        self._untrained_payloads = []

    def search(self, query, top_k=5):
        if not self.is_trained:
            raise RuntimeError("索引尚未训练，无法搜索！")
            
        query = np.array(query, dtype=np.float32)
        
        # ==========================================
        # 💡 IVF 核心提速逻辑
        # ==========================================
        # 1. 找出离 query 最近的 nprobe 个桶（比如总共 1000 个桶，只找最近的 10 个）
        if self.metric == 'l2':
            dists_to_centroids = l2_distance(query, self.centroids)
        else:
            dists_to_centroids = cosine_distance(query, self.centroids)
            
        k_probe = min(self.nprobe, self.num_clusters)
        if k_probe == self.num_clusters:
            top_buckets_indices = np.argsort(dists_to_centroids)
        else:
            # 快速选取最近的桶
            top_indices_unordered = np.argpartition(dists_to_centroids, k_probe - 1)[:k_probe]
            local_sorted = np.argsort(dists_to_centroids[top_indices_unordered])
            top_buckets_indices = top_indices_unordered[local_sorted]
            
        # 2. 只有这几个最近的桶，我们才进去搜索里面的详细数据（这就是提速的关键，避免了 99% 的无效搜索）
        all_distances = []
        all_payloads = []
        
        for b_idx in top_buckets_indices:
            bucket = self.buckets[b_idx]
            if bucket.total_count == 0:
                continue
            # 在桶内部进行局部 Flat 搜索
            b_dists, b_payloads = bucket.search(query, top_k=top_k)
            all_distances.extend(b_dists)
            all_payloads.extend(b_payloads)
            
        if len(all_distances) == 0:
            return [], []
            
        # 3. 对收集上来的所有候选结果进行最终排序
        all_distances = np.array(all_distances)
        final_k = min(top_k, len(all_distances))
        
        if final_k == len(all_distances):
            sorted_indices = np.argsort(all_distances)
            return all_distances[sorted_indices], [all_payloads[i] for i in sorted_indices]
            
        # 依然是用 O(N) 的 argpartition 大法进行极速过滤
        top_k_indices_unordered = np.argpartition(all_distances, final_k - 1)[:final_k]
        top_k_distances_unordered = all_distances[top_k_indices_unordered]
        local_sorted_indices = np.argsort(top_k_distances_unordered)
        final_indices = top_k_indices_unordered[local_sorted_indices]
        
        return all_distances[final_indices], [all_payloads[i] for i in final_indices]
