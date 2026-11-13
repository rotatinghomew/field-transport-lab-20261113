import copy
import json
import tempfile
import unittest
from pathlib import Path


def load(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / name).read_text(encoding='utf-8'))


from field_transport_lab.population import profile, total_variation, inspect


class BusinessTests(unittest.TestCase):
    def test_distance_identity(self):
        self.assertEqual(total_variation({'a': 3, 'b': 2}, {'a': 30, 'b': 20}), 0)

    def test_disjoint(self):
        self.assertEqual(total_variation({'a': 1}, {'b': 1}), 1)

    def test_empty_undefined(self):
        self.assertIsNone(total_variation({}, {'b': 1}))

    def test_missing_weight(self):
        result = profile([{'x': None, 'weight': 4}, {'x': 'a', 'weight': 1}], ['x'], 'weight')
        self.assertEqual(result['x']['missing_weight'], 4)

    def test_negative_weight(self):
        with self.assertRaises(ValueError):
            profile([{'x': 1, 'weight': -1}], ['x'], 'weight')

    def test_target_only_tier_visible(self):
        result = inspect(load('transport.json'))
        self.assertNotIn('entry', result['source']['tier']['counts'])
        self.assertIn('entry', result['target']['tier']['counts'])
