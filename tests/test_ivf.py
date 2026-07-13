import unittest
import numpy as np
from nanovector import IndexIVF

class TestIndexIVF(unittest.TestCase):
    def test_ivf_train_and_search(self):
        # 设有两个聚类桶，搜索时只看最近的 1 个桶 (nprobe=1)
        index = IndexIVF(dim=2, num_clusters=2, nprobe=1, metric='l2')
        
        # 人为制造两个极端隔离的簇 (Clusters)
        # Cluster 1: [1, 1] 附近
        index.add([1.0, 1.0], {"id": 1, "cluster": "A"})
        index.add([1.1, 1.0], {"id": 2, "cluster": "A"})
        index.add([1.0, 1.1], {"id": 3, "cluster": "A"})
        
        # Cluster 2: [100, 100] 附近
        index.add([100.0, 100.0], {"id": 4, "cluster": "B"})
        index.add([101.0, 100.0], {"id": 5, "cluster": "B"})
        index.add([100.0, 101.0], {"id": 6, "cluster": "B"})
        
        # 1. 验证必须先训练才能搜
        with self.assertRaises(RuntimeError):
            index.search([1.0, 1.0])
            
        # 2. 执行训练
        index.train()
        self.assertTrue(index.is_trained)
        
        # 3. 搜索测试
        query = [1.0, 1.0]
        distances, payloads = index.search(query, top_k=2)
        
        # 最近的两个必须是 cluster A 的点
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["cluster"], "A")
        self.assertAlmostEqual(distances[0], 0.0) # 自己距离自己是0

if __name__ == '__main__':
    unittest.main()
