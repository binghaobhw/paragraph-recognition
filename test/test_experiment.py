#!/usr/bin/env python
# coding: utf-8

import codecs
import os
import unittest
from mock import Mock
from experiment import DatasetGenerator, generate_train_data


class TestDatasetGenerator(unittest.TestCase):
    def test_generate(self):
        dataset_filename = 'data/test-set.test'
        label_filename = 'data/label.test'
        dataset_generator = DatasetGenerator(dataset_filename, label_filename)
        dataset_generator.generate(2)
        self.assertTrue(os.path.isfile(dataset_filename))
        self.assertTrue(os.path.isfile(label_filename))
        with codecs.open(label_filename, encoding='utf-8') as f:
            a = 0
            for line in f:
                a += 1
            self.assertEqual(a, 4)
        os.remove(dataset_filename)
        os.remove(label_filename)


class TestTrainData(unittest.TestCase):
    def test_train_data(self):
        mock_method = Mock()
        mock_method.feature_names = [u'pronoun', u'proper_noun', u'noun',
                                     u'verb', u'max_sentence_similarity']
        mock_method.features.return_value = [True, True, False, False, 0.5555]
        dataset_filename = 'data/test-set.test'
        label_filename = 'data/label.test'
        train_set_filename = 'data/train-set.test'
        with codecs.open(dataset_filename, encoding='utf-8', mode='wb') as d:
            d.write(u"Q1:乳房的触诊顺序是什么\n"
                    u"A1:你男女？\n"
                    u"Q2:这和问题有关系吗\n"
                    u"A2:看乳房是啥病\n"
                    u"\n"
                    u"Q3:有没有人知道网店管家这款软件，怎么样？\n"
                    u"A3:可以\n"
                    u"Q4:你知道\n"
                    u"\n"
                    u"Q5:买了一双詹姆斯11涂鸦 但是不知道是不是xdr底的 没有吊牌 现在很担心 只要照片可以看出来是不\n"
                    u"A4:盒子还在吗\n"
                    u"Q6:在的话应该怎么看\n"
                    u"\n")
        with codecs.open(label_filename, encoding='utf-8', mode='wb') as l:
            l.write(u'Q1:0\n'
                    u'Q2:1\n'
                    u'Q3:0\n'
                    u'Q4:1\n'
                    u'Q5:0\n'
                    u'Q6:1\n')
        generate_train_data(mock_method, dataset_filename, label_filename,
                   train_set_filename)
        self.assertTrue(os.path.isfile(dataset_filename))
        self.assertTrue(os.path.isfile(label_filename))
        self.assertTrue(os.path.isfile(train_set_filename))
        with codecs.open(train_set_filename, encoding='utf-8') as f:
            a = 0
            for line in f:
                a += 1
            self.assertEqual(a, 7)
        os.remove(dataset_filename)
        os.remove(label_filename)
        os.remove(train_set_filename)


if __name__ == '__main__':
    unittest.main()
