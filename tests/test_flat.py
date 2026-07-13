import unittest
import numpy as np
from nanovector import IndexFlat

class TestIndexFlat(unittest.TestCase):
    def test_l2_search(self):
        index = IndexFlat(dim=3, metric='l2')
        index.add([1.0, 0.0, 0.0], {"name": "A"})
        index.add([0.0, 1.0, 0.0], {"name": "B"})
        index.add([0.0, 0.0, 1.0], {"name": "C"})
        index.add([1.0, 0.1, 0.0], {"name": "A_near"})
        
        query = [1.0, 0.0, 0.0]
        distances, payloads = index.search(query, top_k=2)
        
        # 预期第一个是 A 自己，距离为 0
        self.assertAlmostEqual(distances[0], 0.0)
        self.assertEqual(payloads[0]["name"], "A")
        
        # 预期第二个是 A_near
        self.assertEqual(payloads[1]["name"], "A_near")
        # 它的 L2 距离的平方应该是 (1-1)^2 + (0.1-0)^2 + (0-0)^2 = 0.01
        self.assertAlmostEqual(distances[1], 0.01)
        
    def test_cosine_search(self):
        index = IndexFlat(dim=2, metric='cosine')
        index.add([1.0, 0.0], {"dir": "x"})
        index.add([0.0, 1.0], {"dir": "y"})
        # [2.0, 0.0] 在方向上和 [1.0, 0.0] 完全一致，Cosine 距离应为 0
        index.add([2.0, 0.0], {"dir": "x_scaled"})
        
        query = [1.0, 0.0]
        distances, payloads = index.search(query, top_k=2)
        
        # 预期前两个的距离都是 0（方向平行）
        self.assertAlmostEqual(distances[0], 0.0)
        self.assertAlmostEqual(distances[1], 0.0)
        
        dirs = [payloads[0]["dir"], payloads[1]["dir"]]
        self.assertIn("x", dirs)
        self.assertIn("x_scaled", dirs)

    def test_add_batch(self):
        index = IndexFlat(dim=2, metric='l2')
        vectors = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        payloads = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        index.add_batch(vectors, payloads)
        self.assertEqual(len(index.vectors), 3)
        
        distances, res_payloads = index.search([2.0, 2.0], top_k=1)
        self.assertEqual(res_payloads[0]["id"], 2)
        self.assertAlmostEqual(distances[0], 0.0)

if __name__ == '__main__':
    unittest.main()
