import unittest
import json
import numpy as np
import torch
from app import app, load_models, prepare_input_data, predict_vqc, predict_qnn

class ChurnPredictorTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        load_models()
        self.sample_input = {
            "creditScore": 600,
            "age": 40,
            "tenure": 5,
            "balance": 50000,
            "numProducts": 2,
            "hasCard": 1,
            "isActive": 1,
            "salary": 60000,
            "geography": "France",
            "gender": "Male"
        }

    def test_index_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_models_page(self):
        response = self.app.get('/models')
        self.assertEqual(response.status_code, 200)

    def test_prepare_input_data(self):
        data = prepare_input_data(self.sample_input)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (1, 13))

    def test_predict_endpoint(self):
        response = self.app.post('/predict',
                                 data=json.dumps(self.sample_input),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertIn('predictions', response_data)

    def test_predict_vqc(self):
        input_data = prepare_input_data(self.sample_input)
        pred = predict_vqc(input_data)
        self.assertIn(pred, [0, 1, None])

    def test_predict_qnn(self):
        input_data = prepare_input_data(self.sample_input)
        pred = predict_qnn(input_data)
        self.assertIn(pred, [0, 1, None])


if __name__ == '__main__':
    unittest.main()