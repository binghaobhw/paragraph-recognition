#!/usr/bin/env python
# coding: utf-8
import json
from unittest import TestCase

from mock import Mock

from method import AnalyzedSentence, configure


class TestAnalyzedSentence(TestCase):
    def test_index(self):
        analyzed_sentence = AnalyzedSentence(
            u'fb070223efa1b565414c30bd4343c56e',
            u'[[[{"id": 0, "cont": "我", "pos": "r", "ne": "O", "parent": 1, "relate": "SBV", "arg": []},{ "id": 1, "cont": "是", "pos": "v", "ne": "O", "parent": -1, "relate": "HED", "arg": [{ "id": 0, "type": "A0", "beg": 0, "end": 0}, { "id": 1, "type": "A1", "beg": 2, "end": 3}]}, { "id": 2, "cont": "中国", "pos": "ns", "ne": "S-Ns", "parent": 3, "relate": "ATT", "arg": []}, { "id": 3, "cont": "人", "pos": "n", "ne": "O", "parent": 1, "relate": "VOB", "arg": []}], [{"cont": "那", "parent": 1, "relate": "ATT", "ne": "O", "pos": "r", "arg": [], "id": 0}, {"cont": "手机", "parent": 2, "relate": "SBV", "ne": "O", "pos": "n", "arg": [], "id": 1}, {"cont": "可以", "parent": -1, "relate": "HED", "ne": "O", "pos": "v", "arg": [], "id": 2}, {"cont": "吗", "parent": 2, "relate": "RAD", "ne": "O", "pos": "u", "arg": [], "id": 3}, {"cont": "？", "parent": 2, "relate": "WP", "ne": "O", "pos": "wp", "arg": [], "id": 4}]]]')
        w = json.loads(u'{"cont": "那", "parent": 1, "relate": "ATT", "ne": "O", "pos": "r", "arg": [], "id": 0}')
        self.assertEqual(analyzed_sentence.index(w), (1, 0))

    def test_has_pronoun(self):
        method_config = {
            'essentials': {
                'third_person_pronoun': 'data/third-person-pronoun.txt',
                'demonstrative_pronoun': 'data/demonstrative-pronoun.txt',
                'cue_word': 'data/cue-word.txt',
                'stop_word': 'data/stop-word.txt'
            },
            'word_similarity_calculators': {
                'how_net': {
                    'class': 'HowNetCalculator',
                    'sememe_tree_file': 'data/whole.dat',
                    'glossary_file': 'data/glossary.dat'
                }
            },
            'sentence_similarity_calculator': {
                'ssc_with_how_net': {
                    'word_similarity_calculator': 'how_net',
                    'score_filename': 'data/how-net-sentence.score'
                }
            },
            'feature_manager': {
                'fm': {
                    'sentence_similarity_calculator': 'ssc_with_how_net'
                }
            },
            'method': {
                'de_boni': {
                    'class': 'DeBoni',
                    'feature_manager': 'fm',
                    'threshold': 0.89,
                    'q_a_threshold': 0.89
                }
            }
        }
        configure(method_config)
        analyzed_sentence = AnalyzedSentence(
            u'fb070223efa1b565414c30bd4343c56e',
            u'[[[{"id": 0, "cont": "我", "pos": "r", "ne": "O", "parent": 1, "relate": "SBV", "arg": []},{ "id": 1, "cont": "是", "pos": "v", "ne": "O", "parent": -1, "relate": "HED", "arg": [{ "id": 0, "type": "A0", "beg": 0, "end": 0}, { "id": 1, "type": "A1", "beg": 2, "end": 3}]}, { "id": 2, "cont": "中国", "pos": "ns", "ne": "S-Ns", "parent": 3, "relate": "ATT", "arg": []}, { "id": 3, "cont": "人", "pos": "n", "ne": "O", "parent": 1, "relate": "VOB", "arg": []}], [{"cont": "那", "parent": 1, "relate": "ATT", "ne": "O", "pos": "r", "arg": [], "id": 0}, {"cont": "手机", "parent": 2, "relate": "SBV", "ne": "O", "pos": "n", "arg": [], "id": 1}, {"cont": "可以", "parent": -1, "relate": "HED", "ne": "O", "pos": "v", "arg": [], "id": 2}, {"cont": "吗", "parent": 2, "relate": "RAD", "ne": "O", "pos": "u", "arg": [], "id": 3}, {"cont": "？", "parent": 2, "relate": "WP", "ne": "O", "pos": "wp", "arg": [], "id": 4}]]]')
        mock_index = Mock()
        mock_index.return_value = (0, 11)
        analyzed_sentence.index = mock_index
        result = analyzed_sentence.has_pronoun()
        self.assertFalse(result)