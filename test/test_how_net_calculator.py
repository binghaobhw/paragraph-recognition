#!/usr/bin/env python
# coding: utf-8
from unittest import TestCase
from method import HowNetCalculator


class TestHowNetCalculator(TestCase):
    def test_load_glossary(self):
        how_net_calculator = HowNetCalculator('../data/glossary.dat',
                                              '../data/WHOLE.DAT')


    def test_calculate(self):
        self.fail()

    def test_calculate_concept_similarity(self):
        self.fail()

    def test_first_independent_sememe_similarity(self):
        self.fail()

    def test_other_independent_sememe_similarity(self):
        self.fail()

    def test_key_value_similarity(self):
        self.fail()

    def test_relation_sememe_similarity(self):
        self.fail()

    def test_symbol_sememe_similarity(self):
        self.fail()

    def test_sememe_similarity(self):
        self.fail()

    def test_sememe_distance(self):
        self.fail()