#!/usr/bin/env python
# coding: utf-8
import codecs
import os
from unittest import TestCase
import cPickle
from mock import Mock
from method import FanYang, FeatureManager


class TestFanYang(TestCase):
    def test_train(self):
        train_set_filename = 'data/train-set.txt'
        with codecs.open(train_set_filename, mode='wb', encoding='utf-8') as f:
            f.write(u'1,1,1,1,0.999,1\n'
                    u'0,0,0,0,0.111,0\n')
        mock_fan_yang = Mock(FanYang)
        mock_fan_yang.train_data_filename = train_set_filename
        mock_fan_yang.feature_names = FanYang.feature_names
        mock_fan_yang.classifier = None
        mock_fan_yang.train = FanYang.train.__get__(mock_fan_yang)
        mock_fan_yang.train()
        self.assertIsNotNone(mock_fan_yang.classifier)
        os.remove(train_set_filename)

    def test_features(self):
        mock_fan_yang = Mock(FanYang)
        mock_fan_yang.features = FanYang.features.__get__(mock_fan_yang)
        mock_fan_yang.feature_names = FanYang.feature_names
        mock_feature_manager = Mock(FeatureManager)
        mock_feature_manager.features.return_value = [
            True, True, True, True, 0.5]
        mock_fan_yang.feature_manager = mock_feature_manager
        features = mock_fan_yang.features(None, None, None)
        self.assertEqual(features[4], 0.5)

    def test_is_follow_up(self):
        mock_fan_yang = Mock(FanYang)
        mock_fan_yang.is_follow_up = FanYang.is_follow_up.__get__(mock_fan_yang)
        mock_fan_yang.classifier.predict.return_value = [True, 'dtype']
        mock_fan_yang.features.return_value = [True, True, True, True, 0.5]
        mock_question = Mock()
        mock_question.md5 = u'71f0404c2c3b9590de252ee22453b127'
        result = mock_fan_yang.is_follow_up(mock_question, None, None)
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
        file_name = 'data/fan-yang.classifier'
        with open(file_name, mode='w') as f:
            cPickle.dump(file_name, f)
        mock_fan_yang.classifier_filename = file_name
        mock_fan_yang.classifier = None
        mock_fan_yang.load_classifier = FanYang.load_classifier.__get__(mock_fan_yang)
        mock_fan_yang.load_classifier()
        self.assertIsNotNone(mock_fan_yang.classifier)
        os.remove(file_name)



