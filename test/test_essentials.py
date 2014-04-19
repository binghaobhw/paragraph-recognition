#!/usr/bin/env python
# coding: utf-8
import unittest
import method


class EssentialsTestCase(unittest.TestCase):
    def setUp(self):
        config = {
            'essentials': {
                'third_person_pronoun': '../data/third-person-pronoun.txt',
                'demonstrative_pronoun': '../data/demonstrative-pronoun.txt',
                'cue_word': '../data/cue-word.txt',
                'stop_word': '../data/stop-word.txt'
            }}
        method.Configurator(config).configure_essentials()
        self.stop_word_dict = method.STOP_WORD_DICT

    def test_stop_word(self):
        self.assertTrue(u'的' in self.stop_word_dict)


if __name__ == '__main__':
    unittest.main()