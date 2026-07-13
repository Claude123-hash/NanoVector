import unittest
import os
import shutil
from nanovector import IndexFlat
from nanovector import save_index_flat, load_index_flat

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_index_data"
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_save_and_load_mmap(self):
        # 1. 创建并保存索引
        index = IndexFlat(dim=2, metric='l2')
        index.add([1.0, 1.0], {"id": 1})
        index.add([2.0, 2.0], {"id": 2})
        index.add([3.0, 3.0], {"id": 3})
        
        save_index_flat(index, self.test_dir)
        
        # 2. 使用 mmap 重新加载
        loaded_index = load_index_flat(self.test_dir, use_mmap=True)
        
        self.assertEqual(loaded_index.total_count, 3)
        self.assertEqual(loaded_index.metric, 'l2')
        self.assertEqual(loaded_index.dim, 2)
        
        # 3. 验证加载后的检索准确性
        distances, payloads = loaded_index.search([2.0, 2.0], top_k=1)
        self.assertAlmostEqual(distances[0], 0.0)
        self.assertEqual(payloads[0]["id"], 2)
        
        # 4. 验证混合架构（在 mmap 的基础上继续新增向量）
        loaded_index.add([4.0, 4.0], {"id": 4})
        self.assertEqual(loaded_index.total_count, 4)
        
        # 检索新增的向量
        distances, payloads = loaded_index.search([4.0, 4.0], top_k=1)
        self.assertAlmostEqual(distances[0], 0.0)
        self.assertEqual(payloads[0]["id"], 4)

if __name__ == '__main__':
    unittest.main()
