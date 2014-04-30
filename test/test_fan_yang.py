#!/usr/bin/env python
# coding: utf-8
import codecs
import os
from unittest import TestCase
from mock import Mock
from method import FanYang, AnalyzedSentence


class TestFanYang(TestCase):
    def test_train(self):
        train_set_filename = 'data/train-set.txt'
        with codecs.open(train_set_filename, mode='wb', encoding='utf-8') as f:
            f.write(u'1,1,1,1,0.999,1\n'
                    u'0,0,0,0,0.111,0\n')
        mock_fan_yang = Mock(FanYang)
        mock_fan_yang.train_data_filename = train_set_filename
        mock_fan_yang.feature_names = FanYang.all_feature_names
        mock_fan_yang.classifier = None
        mock_fan_yang.train = FanYang.train.__get__(mock_fan_yang)
        mock_fan_yang.train()
        self.assertIsNotNone(mock_fan_yang.classifier)

    def test_features(self):
        mock_fan_yang = Mock(FanYang)
        mock_fan_yang.features = FanYang.features.__get__(mock_fan_yang)
        mock_fan_yang.feature_names = FanYang.all_feature_names
        mock_fan_yang.max_sentence_similarity.return_value = 0.5
        question = AnalyzedSentence(u'71f0404c2c3b9590de252ee22453b127',
                                    u'[[[{"cont": "看", "parent": 2, "relate": "SBV", "ne": "O", "pos": "v", "arg": [{"end": 1, "type": "A1", "id": 0, "beg": 1}], "id": 0}, {"cont": "乳房", "parent": 0, "relate": "VOB", "ne": "O", "pos": "n", "arg": [], "id": 1}, {"cont": "是", "parent": -1, "relate": "HED", "ne": "O", "pos": "v", "arg": [{"end": 4, "type": "A1", "id": 0, "beg": 3}], "id": 2}, {"cont": "啥", "parent": 4, "relate": "ATT", "ne": "O", "pos": "r", "arg": [], "id": 3}, {"cont": "病", "parent": 2, "relate": "VOB", "ne": "O", "pos": "n", "arg": [], "id": 4}]]]')
        features = mock_fan_yang.features(question, None, None)
        self.assertTrue(len(features) == 5)
        self.assertEqual(features[4], 0.5)

    def test_is_follow_up(self):
        mock_fan_yang = Mock(FanYang)
        mock_fan_yang.is_follow_up = FanYang.is_follow_up.__get__(mock_fan_yang)
        mock_fan_yang.classifier.predict.return_value = [True, 'dtype']
        mock_fan_yang.max_sentence_similarity.return_value = 0.5
        question = AnalyzedSentence(u'71f0404c2c3b9590de252ee22453b127',
                                    u'[[[{"cont": "看", "parent": 2, "relate": "SBV", "ne": "O", "pos": "v", "arg": [{"end": 1, "type": "A1", "id": 0, "beg": 1}], "id": 0}, {"cont": "乳房", "parent": 0, "relate": "VOB", "ne": "O", "pos": "n", "arg": [], "id": 1}, {"cont": "是", "parent": -1, "relate": "HED", "ne": "O", "pos": "v", "arg": [{"end": 4, "type": "A1", "id": 0, "beg": 3}], "id": 2}, {"cont": "啥", "parent": 4, "relate": "ATT", "ne": "O", "pos": "r", "arg": [], "id": 3}, {"cont": "病", "parent": 2, "relate": "VOB", "ne": "O", "pos": "n", "arg": [], "id": 4}]]]')
        result = mock_fan_yang.is_follow_up(question, None, None)
        self.assertTrue(result)

    def test_save_classifier(self):
        mock_fan_yang = Mock(FanYang)
        mock_fan_yang.classifier_filename = 'data/test-fan-yang.classifier'
        mock_fan_yang.classifier = 'test'
        mock_fan_yang.save_classifier = FanYang.save_classifier.__get__(mock_fan_yang)
        mock_fan_yang.save_classifier()
        self.assertTrue(os.path.isfile(mock_fan_yang.classifier_filename))
        os.remove(mock_fan_yang.classifier_filename)

    def test_load_classifier(self):
        mock_fan_yang = Mock(FanYang)
        mock_fan_yang.classifier_filename = 'data/fan-yang.classifier'
        mock_fan_yang.classifier = None
        mock_fan_yang.load_classifier = FanYang.load_classifier.__get__(mock_fan_yang)
        FanYang.load_classifier(mock_fan_yang)
        self.assertIsNotNone(mock_fan_yang.classifier)



