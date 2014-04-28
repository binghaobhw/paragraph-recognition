#!/usr/bin/env python
# coding: utf-8
from unittest import TestCase
from mock import patch
from method import HowNetCalculator, SememeTreeBuilder, WordConcept

sememe_tree_file = '../data/whole.dat'
glossary_file = '../data/glossary.dat'


class TestSememeTreeBuilder(TestCase):
    def test_build(self):
        SememeTreeBuilder(sememe_tree_file).build()


class TestSememeTree(TestCase):
    def test_path(self):
        sememe_tree = SememeTreeBuilder(sememe_tree_file).build()
        path = sememe_tree.path(sememe_tree[u'become|成为'])
        self.assertTrue(len(path) == 6)


class TestWordConcept(TestCase):
    def test_init(self):
        word = u'CPU'
        concept = u'part|部件,%computer|电脑,heart|心'
        word_concept = WordConcept(word, concept)
        self.assertIsNotNone(word_concept.first_independent_sememe)
        self.assertTrue(len(word_concept.other_independent_sememe) == 1)
        self.assertTrue(len(word_concept.relation_sememe) == 0)
        self.assertTrue(len(word_concept.symbol_sememe) == 1)


class TestHowNetCalculator(TestCase):
    how_net_calculator = HowNetCalculator(sememe_tree_file, glossary_file)
    @patch.object(SememeTreeBuilder, 'build', return_value=None)
    def test_load_glossary(self, mock_build):
        HowNetCalculator(sememe_tree_file, glossary_file)
        mock_build.assert_called()

    def test_sememe_distance(self):
        distance = self.how_net_calculator.sememe_distance(u'become|成为',
                                                           u'own|有')
        self.assertEqual(distance, 5)

    def test_sememe_similarity(self):
        score = self.how_net_calculator.sememe_similarity(u'become|成为',
                                                          u'own|有')
        self.assertGreater(score, 0.0)

    def test_calculate(self):
        score = self.how_net_calculator.calculate(u'吃', u'睡')
        self.assertGreater(score, 0.0)

    def test_calculate_concept_similarity(self):
        concept_a = WordConcept(u'阿法尔语',
                                u'language|语言,#country|国家,ProperName|专')
        concept_b = WordConcept(u'阿富汗',
                                u'place|地方,country|国家,ProperName|专,(Asia|亚洲)')
        score = self.how_net_calculator.calculate_concept_similarity(concept_a,
                                                                     concept_b)
        self.assertGreater(score, 0.0)

    def test_first_independent_sememe_similarity(self):
        sememe_a = u'place|地方'
        sememe_b = u'language|语言'
        score = self.how_net_calculator.first_independent_sememe_similarity(
            sememe_a, sememe_b)
        self.assertGreater(score, 0.0)

    def test_other_independent_sememe_similarity(self):
        list_a = [u'ProperName|专']
        list_b = [u'country|国家', u'ProperName|专', u'(Asia|亚洲)']
        score = self.how_net_calculator.other_independent_sememe_similarity(
            list_a, list_b)
        self.assertGreater(score, 0.0)

    def test_key_value_similarity(self):
        map_a = {u'content': u'beat|打'}
        map_b = {u'manner': u'sorrowful|悲哀'}
        score = self.how_net_calculator.key_value_similarity(map_a, map_b)
        self.assertEqual(score, 0.0)

    def test_relation_sememe_similarity(self):
        map_a = {u'manner': u'happy|福', u'patient': u'aged|老'}
        map_b = {u'manner': u'sorrowful|悲哀'}
        score = self.how_net_calculator.relation_sememe_similarity(map_a, map_b)
        self.assertEqual(score, 0.0)

    def test_symbol_sememe_similarity(self):
        map_a = {u'@': u'sit|坐蹲', u'#': u'livestock|牲畜'}
        map_b = {u'%': u'LandVehicle|车', u'@': u'sit|坐蹲'}
        score = self.how_net_calculator.symbol_sememe_similarity(map_a, map_b)
        self.assertGreater(score, 0.0)
