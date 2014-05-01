#!/usr/bin/env python
# coding: utf-8
from unittest import TestCase
from mock import Mock
from method import build_context, FeatureManager


class TestFeatureManager(TestCase):
    def test_build_context(self):
        context = build_context(None, None, None)
        self.assertDictEqual(context, {'question': None,
                                       'history_questions': None,
                                       'previous_answer': None})

    def test_features(self):
        feature_names = ['pronoun']
        mock_feature_manager = Mock(FeatureManager)
        mock_feature_manager.feature_name_method_map = FeatureManager.\
            feature_name_method_map
        mock_feature_manager.features = FeatureManager.features.__get__(
            mock_feature_manager)
        mock_feature_manager.has_pronoun.return_value = False
        features = mock_feature_manager.features(None, None, None,
                                                 feature_names)
        self.assertTrue(len(features) == 1)
        self.assertFalse(features[0])

    def test_has_pronoun(self):
        self.fail()

    def test_has_cue_word(self):
        self.fail()

    def test_has_proper_noun(self):
        self.fail()

    def test_has_noun(self):
        self.fail()

    def test_has_verb(self):
        self.fail()

    def test_has_same_named_entity(self):
        self.fail()

    def test_has_sbv_or_vob(self):
        self.fail()

    def test_largest_question_similarity(self):
        self.fail()

    def test_qa_similarity(self):
        self.fail()

    def test_largest_similarity(self):
        self.fail()