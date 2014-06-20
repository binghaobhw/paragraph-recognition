#!/usr/bin/env python
# coding: utf-8
import codecs
import os
from unittest import TestCase
import method


class TestConfigurator(TestCase):
    def test_configure_essentials(self):
        essentials = {
            'third_person_pronoun': 'data/third-person-pronoun.txt',
            'demonstrative_pronoun': 'data/demonstrative-pronoun.txt',
            'cue_word': 'data/cue-word.txt',
            'stop_word': 'data/stop-word.txt'
        }
        method.Configurator({'essentials': essentials}).configure_essentials()
        self.assertTrue(u'他' in method.ESSENTIALS_DICT['third_person_pronoun'])
        self.assertTrue(u'这' in method.ESSENTIALS_DICT['demonstrative_pronoun'])
        self.assertTrue(u'其他' in method.ESSENTIALS_DICT['cue_word'])
        self.assertTrue(u"'" in method.ESSENTIALS_DICT['stop_word'])
        self.assertFalse(u"'" in method.ESSENTIALS_DICT['third_person_pronoun'])

    def test_configure_word_similarity_calculator(self):
        word_similarity_calculators = {
            'hi': {
                'class': 'HowNetCalculator',
                'sememe_tree_file': 'sdlf',
                'glossary_file': 'a'}}
        method.Configurator({
            'word_similarity_calculators': word_similarity_calculators}).\
            configure_word_similarity_calculator()
        self.assertIsNotNone(method.word_similarity_calculators['hi'])

    def test_configure_sentence_similarity_calculator(self):
        sentence_similarity_calculator = {
            'ssc': {
                'word_similarity_calculator': 'how_net',
                'score_filename': 'data/sentence-score.cache'
            }
        }
        method.word_similarity_calculators['how_net'] = None
        method.Configurator({
            'sentence_similarity_calculator': sentence_similarity_calculator}).\
            configure_sentence_similarity_calculator()
        self.assertIsNotNone(method.sentence_similarity_calculators['ssc'])

    def test_configure_feature_manager(self):
        feature_manager = {
            'fm': {
                'sentence_similarity_calculator': 'ssc'
            }
        }
        method.sentence_similarity_calculators['ssc'] = None
        method.Configurator({
            'feature_manager': feature_manager}).\
            configure_feature_manager()
        self.assertIsNotNone(method.feature_managers['fm'])

    def test_configure_method(self):
        method_ = {
            'm': {
                'class': 'FanYang',
                'feature_manager': 'fm',
                'train_data_filename': 'tdf'
            }
        }
        method.feature_managers['fm'] = None
        method.Configurator({'method': method_}).configure_method()
        self.assertIsNotNone(method.methods['m'])