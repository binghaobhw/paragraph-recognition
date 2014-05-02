#!/usr/bin/env python
# coding: utf-8
from unittest import TestCase
from mock import Mock
from method import build_context, FeatureManager, SentenceSimilarityCalculator


class TestFeatureManager(TestCase):
    def test_build_context(self):
        context = build_context(None, None, None)
        self.assertDictEqual(context, {'question': None,
                                       'history_questions': None,
                                       'previous_answer': None})

    def test_features(self):
        feature_names = ['pronoun']
        feature_manager = FeatureManager(None)
        mock_question = Mock()
        mock_question.has_pronoun.return_value = False
        features = feature_manager.features(mock_question, None, None,
                                            feature_names)
        mock_question.has_pronoun.assert_called()
        self.assertTrue(len(features) == 1)
        self.assertFalse(features[0])

    def test_has_pronoun(self):
        mock_question = Mock()
        mock_question.has_pronoun.return_value = False
        context = {'question': mock_question}
        result = FeatureManager.has_pronoun(context)
        self.assertFalse(result)

    def test_has_cue_word(self):
        mock_question = Mock()
        mock_question.has_cue_word.return_value = False
        context = {'question': mock_question}
        result = FeatureManager.has_cue_word(context)
        self.assertFalse(result)

    def test_has_proper_noun(self):
        mock_question = Mock()
        mock_question.has_proper_noun.return_value = False
        context = {'question': mock_question}
        result = FeatureManager.has_proper_noun(context)
        self.assertFalse(result)

    def test_has_noun(self):
        mock_question = Mock()
        mock_question.has_noun.return_value = False
        context = {'question': mock_question}
        result = FeatureManager.has_noun(context)
        self.assertFalse(result)

    def test_has_verb(self):
        mock_question = Mock()
        mock_question.has_verb.return_value = False
        context = {'question': mock_question}
        result = FeatureManager.has_verb(context)
        self.assertFalse(result)

    def test_has_same_named_entity(self):
        mock_question = Mock()
        mock_question.named_entities.return_value = iter([{'cont': '1'},
                                                          {'cont': '2'}])
        mock_history_question = Mock()
        mock_history_question.named_entities.return_value = iter([])
        mock_history_questions = [mock_history_question]
        mock_previous_answer = Mock()
        mock_previous_answer.named_entities.return_value = iter([
            {'cont': '1'}])
        context = {'question': mock_question,
                   'history_questions': mock_history_questions,
                   'previous_answer': mock_previous_answer}
        result = FeatureManager.has_same_named_entity(context)
        self.assertTrue(result)

    def test_has_sbv_or_vob(self):
        mock_question = Mock()
        mock_question.has_sbv.return_value = False
        mock_question.has_vob.return_value = True
        context = {'question': mock_question}
        result = FeatureManager.has_sbv_or_vob(context)
        self.assertTrue(result)

    def test_largest_question_similarity(self):
        context = {'question': None,
                   'history_questions': None}
        mock_ssc = Mock(SentenceSimilarityCalculator)
        mock_ssc.max.return_value = False
        feature_manager = FeatureManager(mock_ssc)
        result = feature_manager.largest_question_similarity(context)
        self.assertFalse(result)

    def test_qa_similarity(self):
        context = {'question': None,
                   'previous_answer': None}
        mock_ssc = Mock(SentenceSimilarityCalculator)
        mock_ssc.calculate.return_value = 0.111
        feature_manager = FeatureManager(mock_ssc)
        result = feature_manager.qa_similarity(context)
        self.assertEqual(result, 0.111)

    def test_largest_similarity(self):
        context = {'question': None,
                   'history_questions': [None],
                   'previous_answer': None}
        mock_ssc = Mock(SentenceSimilarityCalculator)
        mock_ssc.max.return_value = 0.111
        feature_manager = FeatureManager(mock_ssc)
        result = feature_manager.largest_similarity(context)
        self.assertEqual(result, 0.111)