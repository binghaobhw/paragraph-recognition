#!/usr/bin/env python
# coding: utf-8
from unittest import TestCase
import method


class TestConfigurator(TestCase):
    def setUp(self):
        config = {
            'essentials': {
                'third_person_pronoun': '../data/third-person-pronoun.txt',
                'demonstrative_pronoun': '../data/demonstrative-pronoun.txt',
                'cue_word': '../data/cue-word.txt',
                'stop_word': '../data/stop-word.txt'
            }}
        method.Configurator(config).configure_essentials()

    def test_configure(self):
        self.fail()

    def test_configure_essentials(self):
        stop_word_dict = method.STOP_WORD_DICT
        self.assertTrue(u'的' in stop_word_dict)

    def test_configure_word_similarity_calculator(self):
        self.fail()

    def test_configure_sentence_similarity_calculator(self):
        self.fail()

    def test_configure_method(self):
        self.fail()