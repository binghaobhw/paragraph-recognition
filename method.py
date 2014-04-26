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

ESSENTIALS_DICT = dict.fromkeys(['third_person_pronoun',
                                 'demonstrative_pronoun',
                                 'cue_word',
                                 'stop_word'], {})

THIRD_PERSON_PRONOUN_DICT = ESSENTIALS_DICT['third_person_pronoun']
DEMONSTRATIVE_PRONOUN_DICT = ESSENTIALS_DICT['demonstrative_pronoun']
CUE_WORD_DICT = ESSENTIALS_DICT['cue_word']
STOP_WORD_DICT = ESSENTIALS_DICT['stop_word']


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
        score = self.do_calculate(text_a, text_b)
        if self.cache:
            self.score_cache[key] = score
            logger.debug('add sentence score into cache: %s', score)
        return score

    def do_calculate(self, text_a, text_b):
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
    glossary = {}
    alpha = 1.6
    beta = [0.5, 0.2, 0.17, 0.13]
    gamma = 0.2
    delta = 0.2

    def __init__(self, sememe_tree_file, glossary_file):
        self.sememe_tree = SememeTreeBuilder().build(sememe_tree_file)
        self.load_glossary(glossary_file)

    def load_glossary(self, glossary_file):
        with codecs.open(glossary_file, encoding='utf-8') as f:
            for line in f:
                key, pos, concept = line.split()
                word_description = WordConcept(key, concept)
                if key in self.glossary:
                    self.glossary[key].append(word_description)
                else:
                    self.glossary[key] = [word_description]

    def calculate(self, word_a, word_b):
        max_score = 0.0
        if word_a in self.glossary and word_b in self.glossary:
            concepts_a = self.glossary[word_a]
            concepts_b = self.glossary[word_b]
            for concept_a in concepts_a:
                for concept_b in concepts_b:
                    score = self.calculate_concept_similarity(concept_a, concept_b)
                    if max_score < score:
                        max_score = score
        return max_score

    def calculate_concept_similarity(self, concept_a, concept_b):
        score = 0.0
        if concept_a.is_function_word == concept_b.is_function_word:
            sim = [
                self.first_independent_sememe_similarity(
                    concept_a.first_basic_sememe, concept_b.first_basic_sememe),
                self.other_independent_sememe_similarity(
                    concept_a.other_basic_sememe, concept_b.other_basic_sememe),
                self.relation_sememe_similarity(concept_a.relation_sememe,
                                                concept_b.relation_sememe),
                self.symbol_sememe_similarity(concept_a.relation_symbol,
                                              concept_b.relation_symbol)]
            product = [sim[0]]
            product.append(product[0] * sim[1])
            product.append(product[1] * sim[2])
            product.append(product[2] * sim[3])
            score = reduce(lambda x, y: x+y, map(lambda (x, y): x*y,
                                                 zip(product, self.beta)))
        return score

    def first_independent_sememe_similarity(self, sememe_a, sememe_b):
        return self.sememe_similarity(sememe_a, sememe_b)

    def other_independent_sememe_similarity(self, list_a, list_b):
        score = 0.0
        if not list_a or not list_b:
            return score
        sememe_score = {}
        pop_sememes = {}
        scores = []
        for sememe_a in list_a:
            for sememe_b in list_b:
                score = self.sememe_similarity(sememe_a, sememe_b)
                sememe_score[(sememe_a, sememe_b)] = score
        while len(sememe_score) > 0:
            max_score = 0.0
            key = None
            for (sememe_a, sememe_b), score in sememe_score.items():
                if sememe_a in pop_sememes or sememe_b in pop_sememes:
                    sememe_score.pop((sememe_a, sememe_b))
                    continue
                if max_score < score:
                    max_score = score
                    key = (sememe_a, sememe_b)
            if key is not None:
                pop_sememes[key[0]] = None
                pop_sememes[key[1]] = None
            scores.append(max_score)
        score = sum(scores) / len(scores)
        return score

    def key_value_similarity(self, map_a, map_b):
        score = 0.0
        if not map_a or not map_b:
            return score
        scores = []
        for key in map_a:
            if key in map_b:
                scores.append(self.sememe_similarity(map_a[key], map_b[key]))
        if scores:
            score = sum(scores) / len(scores)
        return score

    def relation_sememe_similarity(self, map_a, map_b):
        return self.key_value_similarity(map_a, map_b)

    def symbol_sememe_similarity(self, map_a, map_b):
        return self.key_value_similarity(map_a, map_b)

    def sememe_similarity(self, text_a, text_b):
        is_a_specific_word = is_specific_word(text_a)
        is_b_specific_word = is_specific_word(text_b)
        # 两个都是具体词
        if is_a_specific_word and is_b_specific_word:
            return 1.0 if text_a == text_b else 0.0
        # 有一个是具体词
        if is_a_specific_word or is_b_specific_word:
            return self.gamma
        distance = self.sememe_distance(text_a, text_b)
        score = self.alpha / (self.alpha+distance) if distance >= 0 else 0.0
        return score

    def sememe_distance(self, text_a, text_b):
        if text_a not in self.sememe_tree or text_b not in self.sememe_tree:
            return -1
        path_a = self.sememe_tree.path(text_a)
        id_b = text_b.id_
        father_id_b = text_b.father
        distance_b = 0  # b到首个公共节点的距离
        while id_b != father_id_b:
            if id_b in path_a:
                distance_a = path_a.index(id_b)  # a到首个公共节点的距离
                return distance_a + distance_b  # a到b的最短路径
            father_b = self.sememe_tree[father_id_b]
            id_b = father_b.id_
            father_id_b = father_b.father
            distance_b += 1
        if id_b == father_id_b and id_b in path_a:
            return path_a.index(id_b)
        return -1


class WordConcept(object):
    def __init__(self, word, concept):
        self.other_independent_sememe = []
        self.relation_sememe = {}
        self.symbol_sememe = {}
        self.word = word
        if concept.startswith('{'):
            self.is_function_word = True
            content = concept[1: len(concept)-1]
        else:
            self.is_function_word = False
            content = concept
        sememes = content.split(',')
        for sememe in sememes:
            # 关系义原描述式
            s = sememe.split('=')
            if len(s) == 2:
                attribute, value = s
                if not value.startswith('('):
                    value = get_chinese_sememe(value)
                self.relation_sememe[attribute] = value
                continue
            # 基本义原描述式
            if sememe.startswith('(') or sememe[0].isalpha():
                self.other_independent_sememe.append(sememe)
                continue
            else:
                # 关系符号描述式
                symbol = sememe[0]
                description = sememe[1:]
                self.symbol_sememe[symbol] = description
                continue
            self.other_independent_sememe.append(get_chinese_sememe(sememe))
        self.first_independent_sememe = self.other_independent_sememe.pop(0)


def get_english_sememe(sememe):
    return sememe.split('|')[0]


def get_chinese_sememe(sememe):
    return sememe.split('|')[1]

def is_specific_word(text):
    return text.startswith('(')

class Sememe(object):
    def __init__(self, id_, content, father):
        self.id_ = id_
        self.content = content
        self.father = father


class SememeTree(object):
    def __init__(self, list_, dict_):
        self.list_ = list_
        self.dict_ = dict_

    def __contains__(self, text):
        return text in self.dict_

    def path(self, sememe):
        id_ = sememe.id_
        father_id = sememe.father
        path = [sememe]
        while id_ != father_id:
            father = self.list_[father_id]
            path.append(father)
            id_ = father_id
            father_id = father.father
        return path


class SememeTreeBuilder(object):
    def build(self, file_name):
        with codecs.open(file_name, encoding='utf-8') as f:
            sememe_list = []
            sememe_tree = {}
            for line in f:
                id_, content, father = line.split()
                id_ = int(id_)
                father = int(father)
                sememe = Sememe(id_, content, father)
                sememe_list.append(sememe)
                sememe_tree[content] = sememe
        return SememeTree(sememe_list, sememe_tree)


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
    root_node = None

    def __init__(self, sentence_similarity_calculator, train_data_file):
        super(FanYang, self).__init__(sentence_similarity_calculator)
        self.train_data_file = train_data_file

    def train(self):
        pass

    def is_follow_up(self, question, history_questions, previous_answer):
        feature_vector = self.features(question, history_questions,
                                       previous_answer)
        case = [u'pronoun={}'.format(feature_vector[0]),
                u'proper_noun={}'.format(feature_vector[1]),
                u'noun={}'.format(feature_vector[2]),
                u'verb={}'.format(feature_vector[3]),
                u'max_sentence_similarity={}'.format(feature_vector[4])]
        if self.classifier is None:
            self.train()
        result = self.classifier.classify(self.root_node, case)
        for i in result:
            pass

    def features(self, question, history_questions, previous_answer):
        """返回特征值向量"""
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
                d = ESSENTIALS_DICT[k]
                logger.info('start to load %s from %s', k, v)
                for line in f:
                    d[line.strip()] = None
                logger.info('finished loading %s, size=%s', k,
                            len(d))

    def configure_word_similarity_calculator(self):
        config = self.dict_config['word_similarity_calculators']
        for name, kwargs in config.items():
            class_name = kwargs.pop('class')
            class_ = resolve(class_name)
            # word_similarity_calculators[name] = None
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
