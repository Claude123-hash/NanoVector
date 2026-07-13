import time
import numpy as np
from nanovector import IndexFlat
from nanovector import IndexIVF

def generate_data(num_vectors, dim):
    """
    生成模拟的测试数据。
    为了让数据更符合真实世界的“聚簇”特性，我们先生成几个中心点，然后再在周围加高斯噪声散布数据。
    """
    num_clusters = 50
    centroids = np.random.randn(num_clusters, dim)
    
    cluster_assignments = np.random.randint(0, num_clusters, num_vectors)
    noise = np.random.randn(num_vectors, dim) * 0.5
    data = centroids[cluster_assignments] + noise
    return data.astype(np.float32)

def run_benchmark():
    np.random.seed(42) # 固定随机种子，保证每次压测结果一致
    DIM = 64
    NUM_VECTORS = 50000
    NUM_QUERIES = 200
    TOP_K = 10
    
    print(f"[*] NanoVector 极限压测开始...")
    print(f"[*] 维度: {DIM} | 底库规模: {NUM_VECTORS} | 查询次数: {NUM_QUERIES} | Top-K: {TOP_K}")
    
    data = generate_data(NUM_VECTORS, DIM)
    queries = generate_data(NUM_QUERIES, DIM)
    payloads = [{"id": i} for i in range(NUM_VECTORS)]
    
    # =========================================================
    # 测试 1: Flat 暴力检索 (绝对准确，作为基准 Ground Truth)
    # =========================================================
    print("\n--- 1. 暴力检索基准测试 (Flat Index) ---")
    flat_index = IndexFlat(dim=DIM, metric='l2')
    flat_index.add_batch(data, payloads)
    
    ground_truth_ids = []
    
    start_time = time.time()
    for q in queries:
        _, res_payloads = flat_index.search(q, top_k=TOP_K)
        # 记录绝对正确的 Top-K ID 集合，用于计算召回率
        ground_truth_ids.append(set([p["id"] for p in res_payloads]))
        
    flat_time = time.time() - start_time
    flat_qps = NUM_QUERIES / flat_time
    print(f"[OK] Flat 总耗时: {flat_time:.4f}s | QPS (每秒查询数): {flat_qps:.2f}")
    
    # =========================================================
    # 测试 2: IVF 倒排索引 (高速近似检索)
    # =========================================================
    print("\n--- 2. 倒排索引压测 (IVF Index) ---")
    NUM_CLUSTERS = 200 # 将 5 万数据划分为 200 个桶
    ivf_index = IndexIVF(dim=DIM, num_clusters=NUM_CLUSTERS, metric='l2')
    
    for i, vec in enumerate(data):
        ivf_index.add(vec, payloads[i])
        
    print(f"[*] 开始进行 K-Means 聚类训练 (正在将 {NUM_VECTORS} 条数据聚类成 {NUM_CLUSTERS} 个桶)...")
    train_start = time.time()
    ivf_index.train()
    print(f"[OK] 训练完成，耗时: {time.time() - train_start:.2f}s")
    print("\n[*] IVF 性能表现 (权衡 准确度 vs 速度):")
    
    # 测试不同的 nprobe 参数 (搜索时探查的桶数)
    # nprobe 越大：召回率越高，但速度越慢。这是大厂推荐系统的核心权衡！
    for nprobe in [1, 5, 20]:
        ivf_index.nprobe = nprobe
        start_time = time.time()
        
        total_hits = 0
        for i, q in enumerate(queries):
            _, res_payloads = ivf_index.search(q, top_k=TOP_K)
            res_ids = set([p["id"] for p in res_payloads])
            
            # 计算有多少个命中 Ground Truth
            hits = len(res_ids.intersection(ground_truth_ids[i]))
            total_hits += hits
            
        ivf_time = time.time() - start_time
        ivf_qps = NUM_QUERIES / ivf_time
        recall = (total_hits / (NUM_QUERIES * TOP_K)) * 100
        speedup = ivf_qps / flat_qps
        
        print(f"  -> [nprobe={nprobe:<2}] 耗时: {ivf_time:.4f}s | QPS: {ivf_qps:<7.2f} | 召回率(Recall): {recall:>6.2f}% | 速度提升: {speedup:.1f} 倍")

if __name__ == '__main__':
    run_benchmark()
