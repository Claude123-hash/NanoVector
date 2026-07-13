import numpy as np

def l2_distance(query, matrix):
    """
    计算单个 query 向量 (d,) 与 matrix (N, d) 中每一行的 L2 距离的平方。
    为了加快排序速度，我们只计算距离的平方，不进行开方运算。
    """
    # NumPy 广播机制：(d,) 会自动扩展为 (N, d) 进行减法
    diff = query - matrix
    return np.sum(diff ** 2, axis=1)

def cosine_distance(query, matrix):
    """
    计算单个 query 向量 (d,) 与 matrix (N, d) 中每一行的 Cosine 距离。
    Cosine 相似度 = dot(q, x) / (norm(q) * norm(x))
    Cosine 距离 = 1 - Cosine 相似度
    值越小表示方向越相似。
    """
    q_norm = np.linalg.norm(query)
    m_norms = np.linalg.norm(matrix, axis=1)
    
    # 避免除零错误
    q_norm = q_norm if q_norm > 0 else 1e-10
    m_norms = np.where(m_norms == 0, 1e-10, m_norms)
    
    dot_products = np.dot(matrix, query)
    cosine_sim = dot_products / (q_norm * m_norms)
    
    return 1.0 - cosine_sim
