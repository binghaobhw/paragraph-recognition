import unittest
from experiment import DatasetGenerator
import os


class TestDatasetGenerator(unittest.TestCase):
    def test_generate(self):
        dataset_file = 'test-set.txt'
        label_file = 'true-result.txt'
        dataset_generator = DatasetGenerator(dataset_file, label_file)
        dataset_generator.generate()
        os.remove(dataset_file)
        os.remove(label_file)


if __name__ == '__main__':
    unittest.main()
