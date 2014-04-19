#!/usr/bin/env python
# coding: utf-8
import codecs
import json
import logging
import math
import os
import sys
import cPickle as pickle

logger = logging.getLogger('paragraph-recognition.paragraph_recognition')
logger.addHandler(logging.NullHandler())

THIRD_PERSON_PRONOUN_DICT = {}
DEMONSTRATIVE_PRONOUN_DICT = {}
CUE_WORD_DICT = {}
STOP_WORD_DICT = {}
ESSENTIALS_DICT = {'third_person_pronoun_dict': THIRD_PERSON_PRONOUN_DICT,
                   'demonstrative_pronoun_dict': DEMONSTRATIVE_PRONOUN_DICT,
                   'cue_word_dict': CUE_WORD_DICT,
                   'stop_word_dict': STOP_WORD_DICT}

word_similarity_calculators = {}
sentence_similarity_calculators = {}
methods = {}


def vector_cos(a, b):
    if len(a) != len(b):
        raise ValueError('different length: {}, {}'.format(len(a), len(b)))
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        raise ZeroDivisionError()
    else:
        return part_up / part_down


class AnalyzedSentence(object):
    def __init__(self, md5, sentence):
        if isinstance(sentence, unicode):
            self.json = json.loads(sentence)
        elif isinstance(sentence, list):
            self.json = sentence
        else:
            raise TypeError('expecting type is unicode or json, but {}'.
                            format(sentence.__class__))
        self.md5 = md5

    def has_noun(self):
        result = False
        for w in self.words_with_pos_tag('n'):
            result = True
            break
        logger.info('%s', result)
        return result

    def has_proper_noun(self):
        result = False
        for w in self.words():
            if w['ne'] != '0':
                result = True
                break
        logger.info('%s', result)
        return result

    def has_pronoun(self):
        len_threshold = 10
        result = False
        if self.word_count() < len_threshold:
            for pronoun in self.pronouns():
                if pronoun['cont'] in THIRD_PERSON_PRONOUN_DICT \
                        or pronoun['cont'] in DEMONSTRATIVE_PRONOUN_DICT:
                    result = True
                    break
        logger.info('%s', result)
        return result

    def word_count(self):
        return sum(len(s) for s in self.sentences())

    def has_cue_word(self):
        result = False
        for word in self.words():
            if word['cont'] in CUE_WORD_DICT:
                result = True
                break
        logger.info('%s', result)
        return result

    def pronouns(self):
        for r in self.words_with_pos_tag('r'):
            yield r

    def has_verb(self):
        result = False
        for w in self.words_with_pos_tag('v'):
            result = True
            break
        logger.info('%s', result)
        return result

    def words_with_pos_tag(self, x):
        for w in self.words():
            if w['pos'] == x:
                yield w

    def sentences(self):
        for p in self.json:
            for s in p:
                yield s

    def words(self):
        for s in self.sentences():
            for w in s:
                yield w

    def words_exclude_stop(self):
        for w in self.words():
            if w['cont'] not in STOP_WORD_DICT:
                yield w


class SentenceSimilarityCalculator(object):
    score_cache = {}

    def __init__(self, word_similarity_calculator, cache=False,
                 cache_file_name=''):
        self.word_similarity_calculator = word_similarity_calculator
        self.cache = cache
        self.cache_file_name = cache_file_name
        if self.cache and os.path.isfile(self.cache_file_name):
            with open(self.cache_file_name, 'rb') as f:
                logger.info('start to load score cache')
                self.score_cache = pickle.load(f)
                logger.info('finished loading score cache')

    def __del__(self):
        if self.score_cache and self.cache:
            with open(self.cache_file_name, 'wb') as f:
                logger.info('start to save score cache')
                pickle.dump(self.score_cache, f)
                logger.info('finished saving score cache')

    def calculate(self, text_a, text_b):
        if self.cache:
            key = (text_a.md5, text_b.md5)
            if key in self.score_cache:
                score = self.score_cache[key]
                logger.debug('sentence score from cache: %s', score)
                return score
        score = self._calculate(text_a, text_b)
        if self.cache:
            self.score_cache[key] = score
            logger.debug('add sentence score into cache: %s', score)
        return score

    def _calculate(self, text_a, text_b):
        score = 0.0
        text_a_len = 0
        for word in text_a.words_exclude_stop():
            text_a_len += 1
            word_cont = word['cont']
            # word similarity
            max_word_score = 0.0
            for word_to_compare in text_b.words_exclude_stop():
                word_score = self.word_similarity_calculator.calculate(
                    word_cont, word_to_compare['cont'])
                if max_word_score < word_score:
                    max_word_score = word_score
            score += max_word_score
        if text_a_len != 0:
            score /= text_a_len
        logger.debug('sentence score: %s', score)
        return score


class WordSimilarityCalculator(object):
    def __init__(self):
        pass

    def calculate(self, word_a, word_b):
        pass


class HowNetCalculator(WordSimilarityCalculator):
    def __init__(self, word_similarity_table):
        super(HowNetCalculator, self).__init__()
        self.word_similarity_table = word_similarity_table

    def calculate(self, word_a, word_b):
        score = 0.0
        key = word_a, word_b
        if key in self.word_similarity_table:
            score = self.word_similarity_table[key]
        return score


class WordEmbeddingCalculator(WordSimilarityCalculator):
    word_embedding_vectors = {}

    def __init__(self, vector_file_name):
        super(WordEmbeddingCalculator, self).__init__()
        self.vector_file_name = vector_file_name
        self.build_word_embedding_vectors()

    def build_word_embedding_vectors(self):
        with codecs.open(self.vector_file_name, encoding='utf-8') as f:
            logger.info('start to build word vector from %s',
                        self.vector_file_name)
            for line in f:
                line = line.strip()
                columns = line.split(' ')
                word = columns[0]
                vector = [float(num_text) for num_text in columns[1:]]
                self.word_embedding_vectors[word] = vector
            logger.info('finished building word vector')

    def calculate(self, word_a, word_b):
        score = 0.0
        if word_a in self.word_embedding_vectors \
                and word_b in self.word_embedding_vectors:
            raw_score = vector_cos(self.word_embedding_vectors[word_a],
                                   self.word_embedding_vectors[word_b])
            score = (raw_score + 1) / 2
        logger.debug('[%s, %s] score: %s', word_a, word_b, score)
        return score


class AbstractMethod(object):
    def __init__(self, sentence_similarity_calculator):
        self.sentence_similarity_calculator = sentence_similarity_calculator

    def is_follow_up(self, question, history_questions, previous_answer):
        pass

    def train(self, question, history_questions, previous_answer):
        pass

    def save_model(self):
        pass

    def max_sentence_similarity(self, question, history_questions):
        # sentence similarity
        max_sentence_score = 0.0
        for history_question in history_questions:
            score = self.sentence_similarity_calculator.calculate(
                question, history_question)
            if max_sentence_score < score:
                max_sentence_score = score
        logger.info('max sentence score: %s', max_sentence_score)
        return max_sentence_score


class DeBoni(AbstractMethod):

    def __init__(self, sentence_similarity_calculator):
        super(DeBoni, self).__init__(sentence_similarity_calculator)
        self.q_q_threshold = 0.89
        self.q_a_threshold = 0.89

    def is_follow_up(self, question, history_questions, previous_answer):
        follow_up = False
        if question.has_pronoun() or \
                question.has_cue_word() or \
                not question.has_verb() or \
                (history_questions and self.max_sentence_similarity(
                    question, history_questions) > self.q_q_threshold) or \
                (previous_answer and self.sentence_similarity_calculator.
                    calculate(question, previous_answer) > self.q_a_threshold):
            follow_up = True
        return follow_up


class FanYang(AbstractMethod):
    classifier = None

    def __init__(self, sentence_similarity_calculator):
        super(FanYang, self).__init__(sentence_similarity_calculator)

    def train(self, question, history_questions, previous_answer):
        super(FanYang, self).train(question, history_questions,
                                   previous_answer)

    def is_follow_up(self, question, history_questions, previous_answer):
        feature_vector = self.features(question, history_questions, previous_answer)

    def save_model(self):
        super(FanYang, self).save_model()

    def features(self, question, history_questions, previous_answer):
        """返回特征值list"""
        feature_vector = [question.has_pronoun(),
                          question.has_proper_noun(),
                          question.has_noun(),
                          question.has_verb(),
                          self.max_sentence_similarity(question,
                                                       history_questions)]
        return feature_vector


def get_method(name):
    if name in methods:
        return methods[name]
    raise ValueError('no method named {}'.format(name))


class Configurator(object):
    def __init__(self, dict_config):
        self.dict_config = dict_config

    def configure(self):
        word_similarity_calculators.clear()
        methods.clear()
        self.configure_essentials()
        self.configure_word_similarity_calculator()
        self.configure_sentence_similarity_calculator()
        self.configure_method()

    def configure_essentials(self):
        config = self.dict_config['essentials']
        for k, v in config.items():
            with codecs.open(v, encoding='utf-8') as f:
                logger.info('start to load %s from %s', k, v)
                ESSENTIALS_DICT[k] = dict.fromkeys([line.strip() for line in f])
                logger.info('finished loading %s, size=%s', k,
                            len(ESSENTIALS_DICT[k]))

    def configure_word_similarity_calculator(self):
        config = self.dict_config['word_similarity_calculators']
        for name, kwargs in config.items():
            class_name = kwargs.pop('class')
            class_ = resolve(class_name)
            word_similarity_calculators[name] = class_(**kwargs)

    def configure_sentence_similarity_calculator(self):
        config = self.dict_config['sentence_similarity_calculator']
        for name, kwargs in config.items():
            word_calculator_name = kwargs.pop('word_similarity_calculator')
            sentence_similarity_calculators[name] = \
                SentenceSimilarityCalculator(
                    word_similarity_calculators[word_calculator_name],
                    **kwargs)

    def configure_method(self):
        config = self.dict_config['method']
        for name, kwargs in config.items():
            class_name = kwargs.pop('class')
            class_ = resolve(class_name)
            sentence_calculator_name = kwargs.pop(
                'sentence_similarity_calculator')
            methods[name] = class_(
                sentence_similarity_calculators[sentence_calculator_name],
                **kwargs)


def resolve(class_name):
    try:
        if class_name in globals():
            return globals()[class_name]
    except:
        e, tb = sys.exc_info()[1:]
        v = ValueError('cannot resolve %r: %s' % (class_name, e))
        v.__cause__, v.__traceback__ = e, tb
        raise v


def configure(dict_config):
    Configurator(dict_config).configure()
