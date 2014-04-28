#!/usr/bin/env python
# coding: utf-8
from unittest import TestCase
from method import FanYang, AnalyzedSentence, configure, get_method


class TestFanYang(TestCase):
    method_config = {
        'essentials': {
            'third_person_pronoun': '../data/third-person-pronoun.txt',
            'demonstrative_pronoun': '../data/demonstrative-pronoun.txt',
            'cue_word': '../data/cue-word.txt',
            'stop_word': '../data/stop-word.txt'
        },
        'word_similarity_calculators': {
            'how_net': {
                'class': 'HowNetCalculator',
                'sememe_tree_file': '../data/whole.dat',
                'glossary_file': '../data/glossary.dat'
            }
        },
        'sentence_similarity_calculator': {
            'ssc_with_how_net': {
                'cache': True,
                'cache_file_name': '../data/sentence-score-cache-2',
                'word_similarity_calculator': 'how_net'
            }
        },
        'method': {
            'fan_yang': {
                'class': 'FanYang',
                'sentence_similarity_calculator': 'ssc_with_how_net',
                'train_data_file': '../data/train-data.csv'
            }
        }
    }
    configure(method_config)
    fan_yang = get_method('fan_yang')

    def test_train(self):
        fan_yang = FanYang(None, '../data/train-data.csv')
        self.assertIsNotNone(fan_yang.classifier)

    def test_is_follow_up(self):
        question = AnalyzedSentence(u'003ded97d09a05a4b7a48de0737934f0',
                                    u'"arg": [], "id": 1}, {"cont": "这", "parent": 4, "relate": "ATT", "ne": "O", "pos": "r", "arg": [], "id": 2}, {"cont": "两", "parent": 4, "relate": "ATT", "ne": "O", "pos": "m", "arg": [], "id": 3}, {"cont": "个", "parent": 1, "relate": "POB", "ne": "O", "pos": "q", "arg": [], "id": 4}, {"cont": "中选", "parent": 0, "relate": "VOB", "ne": "O", "pos": "v", "arg": [{"end": 4, "type": "LOC", "id": 0, "beg": 1}, {"end": 6, "type": "A1", "id": 1, "beg": 6}], "id": 5}, {"cont": "一个", "parent": 5, "relate": "VOB", "ne": "O", "pos": "m", "arg": [], "id": 6}]]]')
        result = self.fan_yang.is_follow_up(question, None, None)
        self.assertFalse(result)
