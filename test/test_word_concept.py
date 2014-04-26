#!/usr/bin/env python
# coding: utf-8
from unittest import TestCase
from method import WordConcept


class TestWordConcept(TestCase):
    def test_init(self):
        word = u'CPU'
        concept = u'part|部件,%computer|电脑,heart|心'
        word_concept = WordConcept(word, concept)
        self.assertIsNotNone(word_concept.first_independent_sememe)
        self.assertTrue(len(word_concept.other_independent_sememe) == 1)
        self.assertTrue(len(word_concept.relation_sememe) == 0)
        self.assertTrue(len(word_concept.symbol_sememe) == 1)
