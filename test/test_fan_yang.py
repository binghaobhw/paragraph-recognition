#!/usr/bin/env python
# coding: utf-8
from unittest import TestCase
from mock import patch
import method


class TestFanYang(TestCase):

    def test_train(self):
        self.fail()

    @patch.object(method.WordEmbeddingCalculator, 'build_word_embedding_vectors')
    def test_is_follow_up(self, mock_method):
        method_config = {
            'essentials': {
                'third_person_pronoun': '../data/third-person-pronoun.txt',
                'demonstrative_pronoun': '../data/demonstrative-pronoun.txt',
                'cue_word': '../data/cue-word.txt',
                'stop_word': '../data/stop-word.txt'
            },
            'word_similarity_calculators': {
                'word_embedding': {
                    'class': 'WordEmbeddingCalculator',
                    'vector_file_name': '../data/baike-50.vec.txt'
                }
            },
            'sentence_similarity_calculator': {
                'ssc': {
                    'cache': True,
                    'cache_file_name': '../data/sentence-score-cache',
                    'word_similarity_calculator': 'word_embedding'
                }
            },
            'method': {
                'fan_yang': {
                    'class': 'FanYang',
                    'sentence_similarity_calculator': 'ssc',
                    'train_data_file': '../data/train-data.csv'
                }
            }
        }
        method.configure(method_config)
        mock_method.assert_called()
        method_ = method.methods['fan_yang']
        q = method.AnalyzedSentence(u'36b9945374b50b01855ebe582e305583',
                                    u'[[[{"cont": "你", "parent": 1, "relate": "ATT", "ne": "O", "pos": "r", "arg": [], "id": 0}, {"cont": "男女", "parent": -1, "relate": "HED", "ne": "O", "pos": "n", "arg": [], "id": 1}, {"cont": "？", "parent": 1, "relate": "WP", "ne": "O", "pos": "wp", "arg": [], "id": 2}]]]')

        method_.is_follow_up(q, [], None)

    def test_save_model(self):
        self.fail()

    def test_features(self):
        self.fail()