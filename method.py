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
            return True
        return False

    def has_proper_noun(self):
        for w in self.named_entities():
            return True
        return False

    def has_pronoun(self):
        len_threshold = 10
        if self.word_count() < len_threshold:
            for pronoun in self.pronouns():
                if pronoun['cont'] in THIRD_PERSON_PRONOUN_DICT \
                        or pronoun['cont'] in DEMONSTRATIVE_PRONOUN_DICT:
                    return True
        return False

    def word_count(self):
        return sum(len(s) for s in self.sentences())

    def has_cue_word(self):
        for word in self.words():
            if word['cont'] in CUE_WORD_DICT:
                return True
        return False

    def pronouns(self):
        for r in self.words_with_tag('pos', 'r'):
            yield r

    def has_verb(self):
        for w in self.words_with_tag('pos', 'v'):
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
        logger.info('max sentence score: %s', max_sentence_score)
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
        logger.debug('score: %s', score)
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
        logger.debug('[%s, %s] score: %s', word_a, word_b, max_score)
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
        logger.debug('[%s, %s] score: %s', concept_a.word, concept_b.word,
                     score)
        return score

    def first_independent_sememe_similarity(self, sememe_a, sememe_b):
        """Return the first-independent-sememe similarity of two concepts.

        :param sememe_a: unicode
        :param sememe_b: unicode
        :return: float
        """
        score = self.sememe_similarity(sememe_a, sememe_b)
        logger.debug('[%s, %s] score: %s', sememe_a, sememe_b, score)
        return score

    def other_independent_sememe_similarity(self, list_a, list_b):
        """Return the other-independent-sememe similarity of two concepts.

        :param list_a: list[unicode]
        :param list_b: list[unicode]
        :return: float
        """
        score = 0.0
        if not list_a or not list_b:
            logger.debug('one or two of params is None, score: 0.0')
            return score
        sememe_score = {}
        pop_sememes = {}
        scores = []
        for sememe_a in list_a:
            for sememe_b in list_b:
                score = self.sememe_similarity(sememe_a, sememe_b)
                sememe_score[(sememe_a, sememe_b)] = score
        while len(sememe_score) > 0:
            max_score = -1.0
            key = None
            for sememe_tuple, score in sememe_score.items():
                if sememe_tuple[0] in pop_sememes or \
                        sememe_tuple[1] in pop_sememes:
                    sememe_score.pop(sememe_tuple)
                    continue
                if max_score < score:
                    max_score = score
                    key = sememe_tuple
            if key is not None:
                pop_sememes[key[0]] = 1
                pop_sememes[key[1]] = 1
                scores.append(max_score)
        score = sum(scores) / len(scores)
        logger.debug('[%s, %s] score: %s', list_a, list_b, score)
        return score

    def key_value_similarity(self, map_a, map_b):
        """Return the similarity of two key-value maps.

        :param map_a: dict(unicode, unicode)
        :param map_b: dict(unicode, unicode)
        :return: float
        """
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
        """Return the relation-sememe similarity of two concepts.

        :param map_a: dict(unicode, unicode)
        :param map_b: dict(unicode, unicode)
        :return: float
        """
        score = self.key_value_similarity(map_a, map_b)
        logger.debug('[%s, %s] score: %s', map_a, map_b, score)
        return score

    def symbol_sememe_similarity(self, map_a, map_b):
        """Return the symbol-sememe similarity of two concepts.

        :param map_a: dict(unicode, unicode)
        :param map_b: dict(unicode, unicode)
        :return: float
        """
        score = self.key_value_similarity(map_a, map_b)
        logger.debug('[%s, %s] score: %s', map_a, map_b, score)
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
        logger.debug('[%s, %s] score: %s', sememe_a, sememe_b, score)
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

    def __init__(self, vector_file_name):
        super(WordEmbeddingCalculator, self).__init__()
        self.vector_file_name = vector_file_name

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
        if not self.word_embedding_vectors:
            self.build_word_embedding_vectors()
        if word_a in self.word_embedding_vectors \
                and word_b in self.word_embedding_vectors:
            raw_score = vector_cos(self.word_embedding_vectors[word_a],
                                   self.word_embedding_vectors[word_b])
            score = (raw_score + 1) / 2
        logger.debug('[%s, %s] score: %s', word_a, word_b, score)
        return score


class AbstractMethod(object):
    def __init__(self, feature_manager):
        self.feature_manager = feature_manager

    def is_follow_up(self, question, history_questions, previous_answer):
        """Predict whether the question is follow-up.

        :param question: AnalyzedSentence
        :param history_questions: list[AnalyzedSentence]
        :param previous_answer: AnalyzedSentence
        :return: bool
        """
        pass


class DeBoni(AbstractMethod):

    def __init__(self, feature_manager, q_q_threshold,
                 q_a_threshold):
        super(DeBoni, self).__init__(feature_manager)
        self.q_q_threshold = q_q_threshold
        self.q_a_threshold = q_a_threshold

    def is_follow_up(self, question, history_questions, previous_answer):
        follow_up = False
        context = build_context(question, history_questions, previous_answer)
        if self.feature_manager.has_pronoun(context) or \
                self.feature_manager.has_cue_word(context) or \
                not self.feature_manager.has_verb(context) or \
                self.feature_manager.largest_question_similarity(context) > \
                self.q_q_threshold or \
                self.feature_manager.qa_similarity(context) > \
                self.q_a_threshold:
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

    def is_follow_up(self, question, history_questions, previous_answer):
        features = self.features(question, history_questions, previous_answer)
        if not self.classifier:
            if self.classifier_filename and os.path.isfile(
                    self.classifier_filename):
                self.load_classifier()
            elif os.path.isfile(self.train_data_filename):
                self.train()
        predictions = self.classifier.predict([features])
        follow_up = bool(predictions[0])
        logger.debug('question: %s, follow_up: %s', question.md5, follow_up)
        return follow_up

    def features(self, question, history_questions, previous_answer):
        return self.feature_manager.features(
            question, history_questions, previous_answer, self.feature_names)


class ImprovedMethod(FanYang):
    feature_names = [
        'pronoun',
        'cue_word',
        'noun',
        'sbv_or_vob',
        'same_named_entity',
        'word_recurrence_rate',
        'adjacent_question_length_difference',
        'largest_similarity']


def build_context(question, history_questions, previous_answer):
    """Return a question context including question, history and answer.

    :param question: AnalyzedSentence
    :param history_questions: list[AnalyzedSentence]
    :param previous_answer: AnalyzedSentence
    :return: dict
    """
    context = {'question': question,
               'history_questions': history_questions,
               'previous_answer': previous_answer}
    return context


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
            'sbv_or_vob': self.has_sbv_or_vob,
            'same_named_entity': self.has_same_named_entity,
            'largest_similarity': self.largest_similarity,
            'qa_similarity': self.qa_similarity,
            'largest_question_similarity': self.largest_question_similarity,
            'word_recurrence_rate': self.word_recurrence_rate,
            'adjacent_question_length_difference':
                self.adjacent_question_length_difference}

    def features(self, question, history_questiosn, previous_answer,
                 feature_names):
        """Return named features of question.

        :param question: AnalyzedSentence
        :param history_questiosn: list[AnalyzedSentence]
        :param previous_answer: AnalyzedSentence
        :param feature_names: list[unicode]
        :return: list[bool | float]
        """
        features = []
        context = build_context(question, history_questiosn, previous_answer)
        for feature_name in feature_names:
            feature_method = self.feature_name_method_map[feature_name]
            feature = feature_method(context)
            features.append(feature)
        return features

    @staticmethod
    def has_pronoun(context):
        return context['question'].has_pronoun()

    @staticmethod
    def has_cue_word(context):
        return context['question'].has_cue_word()

    @staticmethod
    def has_proper_noun(context):
        return context['question'].has_proper_noun()

    @staticmethod
    def has_noun(context):
        return context['question'].has_noun()

    @staticmethod
    def has_verb(context):
        return context['question'].has_verb()

    @staticmethod
    def has_same_named_entity(context):
        """Determine whether question and its context share same named entity.

        :param context: dict
        :return: bool
        """
        entities = {}
        question = context['question']
        history_questions = context['history_questions']
        previous_answer = context['previous_answer']
        for w in question.named_entities():
            entities[w['cont']] = 1
        if not history_questions:
            return False
        for history_question in history_questions:
            for w in history_question.named_entities():
                if w['cont'] in entities:
                    return True
        if not previous_answer:
            return False
        for w in previous_answer.named_entities():
            if w['cont'] in entities:
                return True
        return False

    @staticmethod
    def has_sbv_or_vob(context):
        """Determine whether question has subject-verb structure.

        :param context: dict
        :return: bool
        """
        question = context['question']
        result = question.has_sbv() or question.has_vob()
        return result

    def largest_question_similarity(self, context):
        """Return largest similarity between question and history questions.

        :param context: dict
        :return: float
        """
        question = context['question']
        history_questions = context['history_questions']
        return self.sentence_similarity_calculator.max(question,
                                                       history_questions)

    def qa_similarity(self, context):
        """Return similarity between question and previous answer.

        :param context: dict
        :return: float
        """
        question = context['question']
        previous_answer = context['previous_answer']
        return self.sentence_similarity_calculator.calculate(question,
                                                             previous_answer)

    def largest_similarity(self, context):
        """Return max similarity between question and context.

        :param context: dict
        :return: float
        """
        question = context['question']
        history_questions = context['history_questions']
        previous_answer = context['previous_answer']
        sentences = []
        if history_questions:
            for i in history_questions:
                sentences.append(i)
            if previous_answer:
                sentences.append(previous_answer)
        return self.sentence_similarity_calculator.max(question, sentences)

    def word_recurrence_rate(self, context):
        """Return rate that words of context recur in current question.

        :param context: dict
        :return: float
        """
        question = context['question']
        word_pool = self.build_word_pool(context)
        if not word_pool:
            return 0.0
        length = 0
        recurrence = 0
        for w in question.words_exclude_stop():
            length += 1
            if w['cont'] in word_pool:
                recurrence += 1
                del word_pool[w['cont']]
        if length > 0 and recurrence > 0:
            return float(recurrence) / length
        else:
            return 0.0

    @staticmethod
    def adjacent_question_length_difference(context):
        """Return length difference between previous question and current question.

        :param context: dict
        :return: float
        """
        question = context['question']
        history_questions = context['history_questions']
        if not history_questions:
            return 0 - question.word_count()
        adjacent_question = history_questions[-1]
        length_difference = adjacent_question.word_count() - \
                            question.word_count()
        return length_difference

    @staticmethod
    def build_word_pool(context):
        """Return a pool containing words of context.

        :param context: dict
        :return: dict
        """
        pool = {}
        history_questions = context['history_questions']
        previous_answer = context['previous_answer']
        if history_questions:
            for i in history_questions:
                for w in i.words_exclude_stop():
                    if w['cont'] not in pool:
                        pool[w['cont']] = 0
        if previous_answer:
            for w in previous_answer.words_exclude_stop():
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
        method_.is_follow_up(question, history_questions, previous_answer)
    """
    Configurator(dict_config).configure()
