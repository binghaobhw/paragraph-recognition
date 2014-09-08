#!/usr/bin/env python
# coding: utf-8
import codecs
import json
import logging
import math
import os
import sys
import cPickle
import re

from sklearn import tree


logger = logging.getLogger('paragraph-recognition.paragraph_recognition')
logger.addHandler(logging.NullHandler())

THIRD_PERSON_PRONOUN_DICT = {}
DEMONSTRATIVE_PRONOUN_DICT = {}
CUE_WORD_DICT = {}
STOP_WORD_DICT = {}

ESSENTIALS_DICT = {'third_person_pronoun': THIRD_PERSON_PRONOUN_DICT,
                   'demonstrative_pronoun': DEMONSTRATIVE_PRONOUN_DICT,
                   'cue_word': CUE_WORD_DICT,
                   'stop_word': STOP_WORD_DICT}

word_similarity_calculators = {}
sentence_similarity_calculators = {}
feature_managers = {}
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
        for w in self.words_with_tag('pos', 'n'):
            logger.debug('%s', w['cont'])
            return True
        return False

    def has_proper_noun(self):
        for w in self.named_entities():
            logger.debug('%s', w['cont'])
            return True
        return False

    def has_pronoun(self):
        len_threshold = 10
        for pronoun in self.pronouns():
            if pronoun['cont'] in THIRD_PERSON_PRONOUN_DICT or \
                    pronoun['cont'] in DEMONSTRATIVE_PRONOUN_DICT:
                index = self.index(pronoun)
                if index[0] == 0 and index[1] <= len_threshold:
                    logger.debug('%s', pronoun['cont'])
                    return True
                else:
                    return False
        return False

    def index(self, word):
        """Return index of word.

        :param word: dict
        :return: (int, int)
        """
        sentence_index = 0
        for s in self.sentences():
            try:
                return sentence_index, s.index(word)
            except ValueError:
                sentence_index += 1
                continue
        return -1, -1

    def word_count(self):
        return sum(len(s) for s in self.sentences())

    def has_cue_word(self):
        for word in self.words():
            if word['cont'] in CUE_WORD_DICT:
                logger.debug('%s', word['cont'])
                return True
        return False

    def pronouns(self):
        for r in self.words_with_tag('pos', 'r'):
            yield r

    def has_verb(self):
        for w in self.words_with_tag('pos', 'v'):
            logger.debug('%s', w['cont'])
            return True
        return False

    def has_sbv(self):
        for w in self.words_with_tag('relate', 'SBV'):
            return True
        return False

    def has_vob(self):
        for w in self.words_with_tag('relate', 'VOB'):
            return True
        return False

    def named_entities(self):
        for w in self.words():
            if w['ne'] != 'O':
                yield w

    def words_with_tag(self, tag, value):
        for w in self.words():
            if w[tag] == value:
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
    scores = {}

    def __init__(self, word_similarity_calculator, score_filename=None):
        self.word_similarity_calculator = word_similarity_calculator
        self.score_filename = score_filename
        self.cache = True if self.score_filename else False

    def __del__(self):
        if self.scores and self.cache:
            with open(self.score_filename, 'wb') as f:
                logger.info('start to save score to %s', self.score_filename)
                cPickle.dump(self.scores, f, cPickle.HIGHEST_PROTOCOL)
                logger.info('finished saving score')

    def calculate(self, a, b):
        """Return the similarity between a and b.

        :param a: AnalyzedSentence
        :param b: AnalyzedSentence
        :return: float
        """
        if not b:
            return 0.0
        if not isinstance(a, AnalyzedSentence):
            raise TypeError('type of a: {}'.format(a.__class__))
        if not isinstance(b, AnalyzedSentence):
            raise TypeError('type of b: {}'.format(b.__class__))
        if self.cache:
            key = (a.md5, b.md5)
            if self.scores:
                if key in self.scores:
                    score = self.scores[key]
                    logger.debug('score from cache: %s, key: %s', score, key)
                    return score
            elif os.path.isfile(self.score_filename):
                with open(self.score_filename, 'rb') as f:
                    logger.info('start to load %s', self.score_filename)
                    self.scores = cPickle.load(f)
                    logger.info('finished loading sentence score')
        score = self.do_calculate(a, b)
        if self.cache:
            self.scores[key] = score
            logger.debug('cache score: %s, key: %s', score, key)
        return score

    def max(self, a, b):
        """Return the largest similarity between a and each sentence of b.

        :param a: AnalyzedSentence
        :param b: list[AnalyzedSentence]
        :return: float
        """
        max_sentence_score = 0.0  # sentence similarity
        if not b:
            return max_sentence_score
        for sentence in b:
            score = self.calculate(a, sentence)
            if max_sentence_score < score:
                max_sentence_score = score
        logger.debug('max sentence similarity: %s', max_sentence_score)
        return max_sentence_score

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
        logger.debug('sentence similarity: %s, text_a: %s, text_b: %s',
                     score, text_a.json, text_b.json)
        return score


class WordSimilarityCalculator(object):
    def __init__(self):
        pass

    def calculate(self, word_a, word_b):
        pass


class HowNetCalculator(WordSimilarityCalculator):
    sememe_tree = None
    glossary = {}
    alpha = 1.6
    beta = [0.5, 0.2, 0.17, 0.13]
    gamma = 0.2
    delta = 0.2

    def __init__(self, sememe_tree_file, glossary_file):
        super(HowNetCalculator, self).__init__()
        self.sememe_tree_file = sememe_tree_file
        self.glossary_file = glossary_file

    def load_glossary(self, glossary_file):
        white_space = re.compile(ur'\s+')
        logger.info('start to load %s', glossary_file)
        with codecs.open(glossary_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                key, pos, concept = white_space.split(line, 2)
                word_description = WordConcept(key, concept)
                if key in self.glossary:
                    self.glossary[key].append(word_description)
                else:
                    self.glossary[key] = [word_description]
        logger.info('finished loading glossary(size=%s)', len(self.glossary))

    def calculate(self, word_a, word_b):
        """Return the similarity of two words.

        :param word_a: unicode
        :param word_b: unicode
        :return: float
        """
        max_score = 0.0
        if not self.glossary:
            self.load_glossary(self.glossary_file)
        if not self.sememe_tree:
            self.sememe_tree = SememeTreeBuilder(self.sememe_tree_file).build()
        if word_a in self.glossary and word_b in self.glossary:
            concepts_a = self.glossary[word_a]
            concepts_b = self.glossary[word_b]
            for concept_a in concepts_a:
                for concept_b in concepts_b:
                    score = self.calculate_concept_similarity(concept_a, concept_b)
                    if max_score < score:
                        max_score = score
        logger.debug('[%s, %s] hownet similarity: %s', word_a, word_b, max_score)
        return max_score

    def calculate_concept_similarity(self, concept_a, concept_b):
        """Return the similarity of two concepts.

        :param concept_a: WordConcept
        :param concept_b: WordConcept
        :return: float
        """
        score = 0.0
        if concept_a.is_function_word == concept_b.is_function_word:
            sim = [
                self.first_independent_sememe_similarity(
                    concept_a.first_independent_sememe,
                    concept_b.first_independent_sememe),
                self.other_independent_sememe_similarity(
                    concept_a.other_independent_sememe,
                    concept_b.other_independent_sememe),
                self.relation_sememe_similarity(concept_a.relation_sememe,
                                                concept_b.relation_sememe),
                self.symbol_sememe_similarity(concept_a.symbol_sememe,
                                              concept_b.symbol_sememe)]
            product = [sim[0]]
            product.append(product[0] * sim[1])
            product.append(product[1] * sim[2])
            product.append(product[2] * sim[3])
            score = reduce(lambda x, y: x+y, map(lambda (x, y): x*y,
                                                 zip(product, self.beta)))
        logger.debug('[%s, %s] concept similarity: %s', concept_a.word,
                     concept_b.word, score)
        return score

    def first_independent_sememe_similarity(self, sememe_a, sememe_b):
        """Return the first-independent-sememe similarity of two concepts.

        :param sememe_a: unicode
        :param sememe_b: unicode
        :return: float
        """
        score = self.sememe_similarity(sememe_a, sememe_b)
        return score

    def other_independent_sememe_similarity(self, list_a, list_b):
        """Return the other-independent-sememe similarity of two concepts.

        :param list_a: list[unicode]
        :param list_b: list[unicode]
        :return: float
        """
        score = 0.0
        if not list_a and not list_b:
            return 1.0
        if not list_a or not list_b:
            return score
        sememe_scores = {}
        pop_sememes = {}
        scores = []
        for sememe_a in list_a:
            for sememe_b in list_b:
                score = self.sememe_similarity(sememe_a, sememe_b)
                sememe_scores[(sememe_a, sememe_b)] = score
        while len(sememe_scores) > 0:
            max_score = -1.0

            key = None
            for sememe_tuple, score in sememe_scores.items():
                if sememe_tuple[0] in pop_sememes or \
                                sememe_tuple[1] in pop_sememes:
                    sememe_scores.pop(sememe_tuple)
                    continue
                if max_score < score:
                    max_score = score
                    key = sememe_tuple
            if key is not None:
                pop_sememes[key[0]] = 1
                pop_sememes[key[1]] = 1
                scores.append(max_score)
        score_num = max(len(list_a), len(list_b))
        while len(scores) < score_num:
            scores.append(self.delta)
        score = sum(scores) / len(scores)
        return score

    def key_value_similarity(self, map_a, map_b):
        """Return the similarity of two key-value maps.

        :param map_a: dict(unicode, unicode)
        :param map_b: dict(unicode, unicode)
        :return: float
        """
        if not map_a and not map_b:
            return 1.0
        if not map_a or not map_b:
            return self.delta
        scores = []
        for key in map_a:
            if key in map_b:
                scores.append(self.sememe_similarity(map_a[key], map_b[key]))
        score_num = max(len(map_a), len(map_b))
        while len(scores) < score_num:
            scores.append(self.delta)
        return sum(scores) / len(scores)

    def relation_sememe_similarity(self, map_a, map_b):
        """Return the relation-sememe similarity of two concepts.

        :param map_a: dict(unicode, unicode)
        :param map_b: dict(unicode, unicode)
        :return: float
        """
        score = self.key_value_similarity(map_a, map_b)
        return score

    def symbol_sememe_similarity(self, map_a, map_b):
        """Return the symbol-sememe similarity of two concepts.

        :param map_a: dict(unicode, unicode)
        :param map_b: dict(unicode, unicode)
        :return: float
        """
        score = self.key_value_similarity(map_a, map_b)
        return score

    def sememe_similarity(self, sememe_a, sememe_b):
        """Return the similarity of two sememes.

        :param sememe_a: unicode
        :param sememe_b: unicode
        :return: float
        """
        is_a_specific_word = is_specific_word(sememe_a)
        is_b_specific_word = is_specific_word(sememe_b)
        # both are specific words
        if is_a_specific_word and is_b_specific_word:
            return 1.0 if sememe_a == sememe_b else 0.0
        # one is specific word
        if is_a_specific_word or is_b_specific_word:
            return self.gamma
        distance = self.sememe_distance(sememe_a, sememe_b)
        score = self.alpha / (self.alpha+distance) if distance >= 0 else 0.0
        return score

    def sememe_distance(self, sememe_a, sememe_b):
        """Return the distance between two sememes.

        :param sememe_a: unicode
        :param sememe_b: unicode
        :return: int
        """
        if sememe_a not in self.sememe_tree or \
                sememe_b not in self.sememe_tree:
            return -1
        sememe_a = self.sememe_tree[sememe_a]
        sememe_b = self.sememe_tree[sememe_b]
        path_a = self.sememe_tree.path(sememe_a)
        id_b = sememe_b.id_
        father_id_b = sememe_b.father
        distance_b = 0  # distance between b and nearest common node
        while id_b != father_id_b:
            if id_b in path_a:
                # distance between a and nearest common node
                distance_a = path_a.index(id_b)
                return distance_a + distance_b  # shortest distance a <-> b
            father_b = self.sememe_tree[father_id_b]
            id_b = father_b.id_
            father_id_b = father_b.father
            distance_b += 1
        if id_b == father_id_b and id_b in path_a:
            return path_a.index(id_b)
        return -1


class WordConcept(object):
    def __init__(self, word, concept):
        self.first_independent_sememe = u''
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
            # relation sememe description
            s = sememe.split('=')
            if len(s) == 2:
                attribute, value = s
                self.relation_sememe[attribute] = value
                continue
            # other independent sememe description
            if sememe.startswith('(') or sememe[0].isalpha():
                self.other_independent_sememe.append(sememe)
                continue
            else:
                # symbol sememe description
                symbol = sememe[0]
                description = sememe[1:]
                self.symbol_sememe[symbol] = description
        if len(self.other_independent_sememe) > 0:
            self.first_independent_sememe = self.other_independent_sememe.pop(0)


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

    def __getitem__(self, text):
        return self.list_[text] if isinstance(text, int) else self.dict_[text]

    def path(self, sememe):
        id_ = sememe.id_
        father_id = sememe.father
        path = [id_]
        while id_ != father_id:
            father = self.list_[father_id]
            id_ = father_id
            father_id = father.father
            path.append(id_)
        return path


class SememeTreeBuilder(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def build(self):
        logger.info('start to load %s', self.file_name)
        with codecs.open(self.file_name, encoding='utf-8') as f:
            sememe_list = []
            sememe_tree = {}
            for line in f:
                id_, content, father = line.split()
                id_ = int(id_)
                father = int(father)
                sememe = Sememe(id_, content, father)
                sememe_list.append(sememe)
                # 跳过已有义原内容
                if content in sememe_tree:
                    continue
                sememe_tree[content] = sememe
        logger.info('finished loading sememe tree(size=%s)', len(sememe_list))
        return SememeTree(sememe_list, sememe_tree)


class WordEmbeddingCalculator(WordSimilarityCalculator):
    word_embedding_vectors = {}

    def __init__(self, vector_filename):
        super(WordEmbeddingCalculator, self).__init__()
        self.vector_filename = vector_filename

    def build_word_embedding_vectors(self):
        with codecs.open(self.vector_filename, encoding='utf-8') as f:
            logger.info('start to build word vector from %s',
                        self.vector_filename)
            for line in f:
                line = line.strip()
                columns = line.split(' ')
                word = columns[0]
                vector = [float(num_text) for num_text in columns[1:]]
                self.word_embedding_vectors[word] = vector
            logger.info('finished building word vector')

    def calculate(self, word_a, word_b):
        score = 0.0
        if not self.word_embedding_vectors:
            self.build_word_embedding_vectors()
        if word_a in self.word_embedding_vectors \
                and word_b in self.word_embedding_vectors:
            raw_score = vector_cos(self.word_embedding_vectors[word_a],
                                   self.word_embedding_vectors[word_b])
            score = (raw_score + 1) / 2
        logger.debug('[%s, %s] word embedding similarity: %s',
                     word_a, word_b, score)
        return score


class AbstractMethod(object):

    def __init__(self, feature_manager, feature_names):
        self.feature_manager = feature_manager
        self.feature_names = feature_names

    def is_follow_up(self, sentence, history_sentences, special=None):
        """Predict whether the sentence is follow-up.

        :param sentence: AnalyzedSentence
        :param history_sentences: list[AnalyzedSentence]
        :param special: AnalyzedSentence
        :return: bool
        """
        pass

    def features(self, sentence, history_sentences, special):
        return self.feature_manager.features(sentence, history_sentences, None,
                                             special)


class DeBoni(AbstractMethod):

    def __init__(self, feature_manager, q_q_threshold,
                 q_a_threshold):
        feature_names = [
            'pronoun',
            'cue_word',
            'verb',
            'largest_question_similarity',
            'qa_similarity'
        ]
        super(DeBoni, self).__init__(feature_manager, feature_names)
        self.q_q_threshold = q_q_threshold
        self.q_a_threshold = q_a_threshold

    def is_follow_up(self, sentence, history_sentences, special=None):
        follow_up = False
        features = self.features(sentence, history_sentences, None)
        if features and len(features) == 5 and (
                features[0] or
                features[1] or
                not features[2] or
                features[3] > self.q_q_threshold or
                features[4] > self.q_a_threshold):
            follow_up = True
        return follow_up


def field_to_right_type(fields):
    """Convert field into right type.

    :param fields: list[unicode]
    :return: list[bool | float]
    """
    result = []
    for field in fields:
        if field.isdigit():
            result.append(bool(int(field)))
        else:
            result.append(float(field))
    return result


class FanYang(AbstractMethod):
    classifier = None
    feature_names = [
        'pronoun',
        'proper_noun',
        'noun',
        'verb',
        'largest_question_similarity']

    def __init__(self, feature_manager, train_data_filename,
                 classifier_filename=None):
        super(FanYang, self).__init__(feature_manager)
        self.train_data_filename = train_data_filename
        self.classifier_filename = classifier_filename

    def __del__(self):
        if self.classifier and self.classifier_filename:
            self.save_classifier()

    def load_classifier(self):
        with open(self.classifier_filename, 'rb') as f:
            logger.info('start to load %', self.classifier_filename)
            self.classifier = cPickle.load(f)
            logger.info('finished load classifier')

    def save_classifier(self):
        with open(self.classifier_filename, 'wb') as f:
            logger.info('start to save classifier to %s',
                        self.classifier_filename)
            cPickle.dump(self.classifier, f, cPickle.HIGHEST_PROTOCOL)
            logger.info('finished save classifier')

    def train(self, max_depth=3, min_samples_leaf=5):
        features = []
        labels = []
        logger.info('start to train %s', self.train_data_filename)
        with codecs.open(self.train_data_filename, encoding='utf-8') as f:
            f.next()
            for line in f:
                line = line.strip()
                fields = line.split(',', len(self.feature_names))
                fields = field_to_right_type(fields)
                features.append(fields[:-1])
                labels.append(fields[-1])
        self.classifier = tree.DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf).\
            fit(features, labels)
        logger.info('finished training')

    def is_follow_up(self, sentence, history_sentences, special=None):
        features = self.features(sentence, history_sentences, None)
        if not self.classifier:
            if self.classifier_filename and os.path.isfile(
                    self.classifier_filename):
                self.load_classifier()
            elif os.path.isfile(self.train_data_filename):
                self.train()
        predictions = self.classifier.predict([features])
        follow_up = bool(predictions[0])
        logger.info('follow_up: %s, sentence: %s', follow_up, sentence.md5)
        return follow_up


class ImprovedMethod(FanYang):
    feature_names = [
        'pronoun',
        'cue_word',
        'noun',
        'sbv_and_vob',
        'same_named_entity',
        'word_recurrence_rate',
        'adjacent_sentence_length_difference',
        'largest_similarity']


def build_context(sentence, history_sentences, special):
    """Return a sentence context including sentence, history.

    :param sentence: AnalyzedSentence
    :param history_sentences: list[AnalyzedSentence]
    :param special: AnalyzedSentence
    :return: dict
    """
    context = {'sentence': sentence,
               'history_sentences': history_sentences,
               'special': special}
    return context


def length_difference(a, b):
        """Return length difference between a and b.

        :param a: AnalyzedSentence
        :param b: AnalyzedSentence
        :return: int
        """
        a_valid = isinstance(a, AnalyzedSentence)
        b_valid = isinstance(b, AnalyzedSentence)
        a_count = a.word_count()
        b_count = b.word_count()
        if a_valid:
            return a_count - b_count if b_valid else a_count
        else:
            return 0 - b_count if b_valid else 0


class FeatureManager(object):
    def __init__(self, sentence_similarity_calculator):
        super(FeatureManager, self).__init__()
        self.sentence_similarity_calculator = sentence_similarity_calculator
        self.feature_name_method_map = {
            'pronoun': self.has_pronoun,
            'cue_word': self.has_cue_word,
            'proper_noun': self.has_proper_noun,
            'noun': self.has_noun,
            'verb': self.has_verb,
            'sbv_and_vob': self.has_sbv_and_vob,
            'same_named_entity': self.has_same_named_entity,
            'largest_similarity': self.largest_similarity,
            'word_recurrence_rate': self.word_recurrence_rate,
            'adjacent_sentence_length_difference': self.adjacent_sentence_length_difference
        }

    def features(self, sentence, history_sentences, special, feature_names):
        """Return named features of sentence.

        :param sentence: AnalyzedSentence
        :param history_sentences: List[AnalyzedSentence]
        :param special: AnalyzedSentence
        :param feature_names: list[unicode]
        :return: list[bool | float]
        """
        features = []
        context = build_context(sentence, history_sentences, special)
        for feature_name in feature_names:
            feature_method = self.feature_name_method_map[feature_name]
            feature = feature_method(context)
            features.append(feature)
        logger.info('%s', features)
        return features

    @staticmethod
    def has_pronoun(context):
        return context['sentence'].has_pronoun()

    @staticmethod
    def has_cue_word(context):
        return context['sentence'].has_cue_word()

    @staticmethod
    def has_proper_noun(context):
        return context['sentence'].has_proper_noun()

    @staticmethod
    def has_noun(context):
        return context['sentence'].has_noun()

    @staticmethod
    def has_verb(context):
        return context['sentence'].has_verb()

    @staticmethod
    def has_sbv_and_vob(context):
        """Determine whether sentence has subject-verb and verb-object structure.

        :param context: dict
        :return: bool
        """
        sentence = context['sentence']
        result = sentence.has_sbv() and sentence.has_vob()
        return result

    @staticmethod
    def has_same_named_entity(context):
        """Determine whether sentence and its context share same named entity.

        :param context: dict
        :return: bool
        """
        entities = {}
        sentence = context['sentence']
        history_sentences = context['history_sentences']
        for w in sentence.named_entities():
            entities[w['cont']] = 1
        if not history_sentences:
            return False
        for history_sentence in history_sentences:
            for w in history_sentence.named_entities():
                if w['cont'] in entities:
                    logger.debug('%s', w['cont'])
                    return True
        return False

    def largest_similarity(self, context):
        """Return max similarity between sentence and context.

        :param context: dict
        :return: float
        """
        sentence = context['sentence']
        history_sentences = context['history_sentences']
        return self.sentence_similarity_calculator.max(sentence, history_sentences)

    def word_recurrence_rate(self, context):
        """Return rate that words of context recur in current sentence.

        :param context: dict
        :return: float
        """
        sentence = context['sentence']
        word_pool = build_word_pool(context['history_sentences'])
        if not word_pool:
            return 0.0
        length = 0
        recurrence = 0
        for w in sentence.words_exclude_stop():
            length += 1
            if w['cont'] in word_pool:
                recurrence += 1
                del word_pool[w['cont']]
        if length > 0 and recurrence > 0:
            return float(recurrence) / length
        else:
            return 0.0

    @staticmethod
    def adjacent_sentence_length_difference(context):
        """Return length difference between previous sentence and current sentence.

        :param context: dict
        :return: int
        """
        sentence = context['sentence']
        history_sentences = context['history_sentences']
        special = context['special']
        if special:
            return length_difference(special, sentence)
        if history_sentences:
            return length_difference(history_sentences[-1], sentence)
        return length_difference(sentence, None)


def build_word_pool(sentences):
        """Return a pool containing words of sentences.

        :param sentences: list[AnalyzedSentence]
        :return: dict
        """
        pool = {}
        if sentences:
            for i in sentences:
                for w in i.words_exclude_stop():
                    if w['cont'] not in pool:
                        pool[w['cont']] = 0
        return pool


def get_method(name):
    if name in methods:
        return methods[name]
    raise ValueError('no method named {}'.format(name))


class Configurator(object):
    def __init__(self, dict_config):
        self.dict_config = dict_config

    def configure(self):
        methods.clear()
        sentence_similarity_calculators.clear()
        word_similarity_calculators.clear()
        self.configure_essentials()
        self.configure_word_similarity_calculator()
        self.configure_sentence_similarity_calculator()
        self.configure_feature_manager()
        self.configure_method()

    def configure_essentials(self):
        config = self.dict_config['essentials']
        for k, v in config.items():
            with codecs.open(v, encoding='utf-8') as f:
                d = ESSENTIALS_DICT[k]
                logger.info('start to load %s from %s', k, v)
                for line in f:
                    d[line.strip()] = 1
                logger.info('finished loading %s, size=%s', k,
                            len(d))

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

    def configure_feature_manager(self):
        config = self.dict_config['feature_manager']
        for name, kwargs in config.items():
            sentence_calculator_name = kwargs.pop(
                'sentence_similarity_calculator')
            feature_managers[name] = FeatureManager(
                sentence_similarity_calculators[sentence_calculator_name])

    def configure_method(self):
        config = self.dict_config['method']
        for name, kwargs in config.items():
            class_name = kwargs.pop('class')
            class_ = resolve(class_name)
            feature_manager_name = kwargs.pop(
                'feature_manager')
            methods[name] = class_(feature_managers[feature_manager_name],
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
    """Configure methods.

    :param dict_config: dict

    example:
        method_config = {
            'essentials': {
                'third_person_pronoun': 'data/third-person-pronoun.txt',
                'demonstrative_pronoun': 'data/demonstrative-pronoun.txt',
                'cue_word': 'data/cue-word.txt',
                'stop_word': 'data/stop-word.txt'
            },
            'word_similarity_calculators': {
                'how_net': {
                    'class': 'HowNetCalculator',
                    'sememe_tree_file': 'data/whole.dat',
                    'glossary_file': 'data/glossary.dat'
                }
            },
            'sentence_similarity_calculator': {
                'ssc_with_how_net': {
                    'word_similarity_calculator': 'how_net',
                    'score_filename': 'data/sentence-score.cache'
                }
            },
            'method': {
                'fan_yang': {
                    'class': 'FanYang',
                    'sentence_similarity_calculator': 'ssc_with_how_net',
                    'train_data_filename': 'data/train-set.txt',
                    'classifier_filename': 'data/fan-yang.classifier'
                }
            }
        }
        configure(method_config)
        method_ = get_method('fan_yang')
        method_.is_follow_up(sentence, history_sentences, previous_answer)
    """
    Configurator(dict_config).configure()
