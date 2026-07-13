import numpy as np
from .distance import l2_distance, cosine_distance

def kmeans_train(X, num_clusters, max_iter=100, metric='l2', tol=1e-4):
    """
    使用 K-Means 将矩阵 X 划分为 num_clusters 个聚类。
    :param X: 训练数据 (N, dim)
    :param num_clusters: 期望的聚类数
    :return: centroids (num_clusters, dim)
    """
    N, dim = X.shape
    if N < num_clusters:
        raise ValueError(f"数据点({N})少于聚类数({num_clusters})，无法训练")
        
    # 1. 初始化：随机抽取样本作为初始的聚类中心点 (Centroids)
    indices = np.random.choice(N, num_clusters, replace=False)
    centroids = X[indices].copy()
    
    for iteration in range(max_iter):
        # 2. 分配聚类 (Assignment Step)
        # 计算所有点到所有中心点的距离
        dists = np.zeros((N, num_clusters), dtype=np.float32)
        
        for k in range(num_clusters):
            centroid = centroids[k]
            if metric == 'l2':
                dists[:, k] = l2_distance(centroid, X)
            else:
                dists[:, k] = cosine_distance(centroid, X)
                
        # assignments: 每个点对应的最近中心点的索引 [N]
        assignments = np.argmin(dists, axis=1)
        
        # 3. 更新中心点 (Update Step)
        new_centroids = np.zeros_like(centroids)
        max_diff = 0.0
        
        for k in range(num_clusters):
            # 找出所有属于第 k 个聚类的点
            cluster_points = X[assignments == k]
            
            if len(cluster_points) > 0:
                # 求平均值作为新的中心点
                new_centroid = np.mean(cluster_points, axis=0)
                if metric == 'cosine':
                    # 余弦距离场景下，中心点需要归一化，保持方向特征
                    norm = np.linalg.norm(new_centroid)
                    if norm > 0:
                        new_centroid /= norm
            else:
                # 极端情况：某个中心点周围没有任何点（空桶），重新随机挑一个点复活
                new_centroid = X[np.random.choice(N)]
                
            diff = np.linalg.norm(centroids[k] - new_centroid)
            max_diff = max(max_diff, diff)
            new_centroids[k] = new_centroid
            
        centroids = new_centroids
        
        # 如果中心点不再剧烈移动，说明算法收敛，可以提前结束
        if max_diff < tol:
            break
            
    return centroids
