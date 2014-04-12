#!/usr/bin/env python
# coding: utf-8
from Queue import Queue
import codecs
import getopt
import hashlib
import json
import logging
import logging.config
import math
import os
import requests
import sys
from data_access import (Session,
                         Paragraph,
                         LtpResult)
from log_config import (LOG_PROJECT_NAME,
                        LOGGING)

logger = logging.getLogger(LOG_PROJECT_NAME)
LTP_URL = 'http://api.ltp-cloud.com/analysis'
API_KEY = 'u1Q1k8U6tglHca7ZZJ6qTBaq2k0QYwyXNqyE3kVu'
FORMAT = 'json'
PATTERN = 'all'
param = {'api_key': API_KEY,
         'format': FORMAT,
         'pattern': PATTERN,
         'text': None}

WORD_VECTOR = {}

PUNCTUATION_TABLE = [u' ', u'.', u'。', u',', u'，', u'!', u'！', u';', u'；',
                     u'﹖', u'?', u'？', u'～', u'~']

THIRD_PERSON_PRONOUN_DICT = dict.fromkeys([u'他', u'她', u'它', u'他们',
                                           u'她们', u'它们',])

DEMONSTRATIVE_PRONOUN_DICT = dict.fromkeys([u'这', u'这儿', u'这么', u'这里',
                                            u'这会儿', u'这样', u'这么样',
                                            u'这些', u'那', u'那儿', u'那么',
                                            u'那里', u'那会儿', u'那样',
                                            u'那么样', u'那些'])


with codecs.open('data/cue-word.txt', encoding='utf-8', mode='rb') as f:
    CUE_WORD_DICT = dict.fromkeys([line.strip() for line in f])


with codecs.open('data/stop-word.txt', encoding='utf-8', mode='rb') as f:
    STOP_WORD_DICT = dict.fromkeys([line.strip() for line in f])


def build_word_vector():
    with codecs.open('data/baike-50.vec.txt', encoding='utf-8', mode='rb') as f:
        logger.info('start to build word vector')
        f.next()
        for line in f:
            line = line.strip()
            columns = line.split(' ')
            word = columns[0]
            vector = [float(num_text) for num_text in columns[1:]]
            WORD_VECTOR[word] = vector
        logger.info('finished building word vector')


def vector_cos(a, b):
    if len(a) != len(b):
        raise ValueError('different length: {}, {}'.format(len(a), len(b)))
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        raise ZeroDivisionError()
    else:
        return part_up / part_down


def build_param(text):
    param['text'] = text
    return param


def truncate(text):
    for punctuation in PUNCTUATION_TABLE:
        index = text.find(punctuation)
        if 0 < index < 50:
            return text[:index]
    return text[:50]


def analyze(text):
    response = requests.get(LTP_URL, params=build_param(text), timeout=60)
    if not response.ok:
        if response.status_code == 400 and response.json()['error_message'] == 'SENTENCE TOO LONG':
            logger.info('sentence too long, truncate')
            truncated_text = truncate(text)
            response = requests.get(LTP_URL, params=build_param(truncated_text), timeout=60)
        else:
            raise RuntimeError('bad response code={} url={} text={}'.format(
                response.status_code, response.url, response.text))
    return response.json()


class AnalyzedResult(object):
    def __init__(self, result):
        if isinstance(result, unicode):
            self.json = json.loads(result)
        elif isinstance(result, list):
            self.json = result
        else:
            raise TypeError('expecting type is unicode or json, but {}'.
                            format(result.__class__))

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


def md5(text):
    data = text.encode('utf-8') if isinstance(text, unicode) else text
    return hashlib.md5(data).hexdigest()


def get_analyzed_result(question_text):
    if question_text is None:
        return None
    md5_string = md5(question_text)
    ltp_result = Session.query(LtpResult).filter_by(md5=md5_string).first()
    if ltp_result is not None:
        analyzed_result = AnalyzedResult(ltp_result.json_text)
    else:
        try:
            result_json = analyze(question_text)
        except RuntimeError:
            logger.error('fail to invoke ltp api, text=%s', question_text, exc_info=True)
            raise RuntimeError()

        save_analyzed_result(md5_string, result_json)
        analyzed_result = AnalyzedResult(result_json)
    return analyzed_result


def save_analyzed_result(md5_string, result_json):
    ltp_result = LtpResult(md5_string,
                           json.dumps(result_json, ensure_ascii=False))
    Session.add(ltp_result)
    logger.info('start to insert ltp result, md5=%s', md5_string)
    try:
        Session.commit()
    except:
        Session.rollback()
        logger.error('fail to insert')
    logger.info('finished inserting ltp result')


class DataSetGenerator():
    def __init__(self, data_set_file_name='data/test-set.txt',
                 result_file_name='data/actual-result.txt'):
        self.result_pattern = u'{}{}:{}\n'
        self.queue = Queue()
        self.question_num = 1
        self.answer_num = 1
        self.data_set_file_name = data_set_file_name
        self.result_file_name = result_file_name

    def generate_paragraph(self, paragraph):
        paragraph_lines = [self.result_pattern.format('Q', self.question_num, paragraph.question.title)]
        result_lines = [self.result_pattern.format('Q', self.question_num, 'N')]
        self.question_num += 1
        for reply in paragraph.replies:
            if reply.is_deleted == 1:
                continue
            if reply.is_question():
                test_line = self.result_pattern.format('Q',  self.question_num, reply.content)
                result_line = self.result_pattern.format('Q', self.question_num, 'F')
                result_lines.append(result_line)
                self.question_num += 1
            else:
                test_line = self.result_pattern.format('A', self.answer_num, reply.content)
                self.answer_num += 1
            paragraph_lines.append(test_line)
        return paragraph_lines, result_lines

    def generate(self):
        previous_category_id = None
        with codecs.open(self.data_set_file_name, encoding='utf-8', mode='wb') as data_set:
            with codecs.open(self.result_file_name, encoding='utf-8', mode='wb') as result:
                logger.info('start to generate data set')
                for paragraph in Session.query(Paragraph).filter(Paragraph.paragraph_id <= 400).filter(Paragraph.is_deleted == 0).all():
                    # 当前分类与上一话段分类一样，先不写，入队列
                    if paragraph.question.category_id == previous_category_id:
                        logger.info('same category id, put %s into queue',
                                    paragraph.paragraph_id)
                        self.queue.put(paragraph)
                        continue
                    paragraph_lines, result_lines = self.generate_paragraph(paragraph)
                    paragraph_lines.append('\n')
                    data_set.writelines(paragraph_lines)
                    result.writelines(result_lines)
                    previous_category_id = paragraph.question.category_id
                    # 看队列头是否能写
                    if not self.queue.empty():
                        paragraph_in_queue = self.queue.get()
                        if paragraph_in_queue.question.category_id == previous_category_id:
                            logger.info('same category id again, put %s into queue', paragraph_in_queue.paragraph_id)
                            self.queue.put(paragraph_in_queue)
                        else:
                            logger.info('dequeue')
                            paragraph_lines, result_lines = self.generate_paragraph(paragraph_in_queue)
                            paragraph_lines.append('\n')
                            data_set.writelines(paragraph_lines)
                            result.writelines(result_lines)
                            previous_category_id = paragraph_in_queue.question.category_id
                while not self.queue.empty():
                    paragraph_in_queue = self.queue.get()
                    paragraph_lines, result_lines = self.generate_paragraph(paragraph_in_queue)
                    paragraph_lines.append('\n')
                    data_set.writelines(paragraph_lines)
                    result.writelines(result_lines)
                logger.info('finished generating data set')


def word_similarity(a, b):
    score = 0.0
    if a in WORD_VECTOR and b in WORD_VECTOR:
        row_score = vector_cos(WORD_VECTOR[a], WORD_VECTOR[b])
        score = (row_score + 1) / 2
    return score


class CalculatorConfigurator(object):
    def __init__(self, dict_config):
        self.dict_config = dict_config

    def config(self):
        pass


class SentenceSimilarityCalculator(object):
    def __init__(self, word_similarity_calculator):
        self.word_similarity_calculator = word_similarity_calculator

    def calculate(self, text_a, text_b):
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
        logger.debug('sentence score=%s', score)
        return score


class WordSimilarityCalculator(object):
    def __init__(self):
        pass

    def calculate(self, word_a, word_b):
        pass


class HowNetCalculator(WordSimilarityCalculator):
    def __init__(self, word_similarity_table):
        self.word_similarity_table = word_similarity_table

    def calculate(self, word_a, word_b):
        score = 0.0
        key = word_a, word_b
        if key in self.word_similarity_table:
            score = self.word_similarity_table[key]
        return score


class WordEmbeddingCalculator(WordSimilarityCalculator):
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors

    def calculate(self, word_a, word_b):
        score = 0.0
        if word_a in self.word_vectors and word_b in self.word_vectors:
            raw_score = vector_cos(self.word_vectors[word_a], self.word_vectors[word_b])
            score = (raw_score + 1) / 2
        return score


def sentence_similarity(text, text_list):
    # sentence similarity
    max_sentence_score = 0.0
    if text is not None and text_list is not None:
        t_list = text_list if isinstance(text_list, list) else [text_list]
        for text_to_compare in t_list:
            sentence_score = 0.0
            sentence_len = 0
            for word in text.words_exclude_stop():
                sentence_len += 1
                word_text = word['cont']
                # word similarity
                max_word_score = 0.0
                for word_to_compare in text_to_compare.words_exclude_stop():
                    word_score = word_similarity(word_text, word_to_compare['cont'])
                    if max_word_score < word_score:
                        max_word_score = word_score
                sentence_score += max_word_score
            if sentence_len != 0:
                sentence_score /= sentence_len
            logger.debug('sentence score=%s', sentence_score)
            if max_sentence_score < sentence_score:
                max_sentence_score = sentence_score
    logger.info('max sentence score=%s', max_sentence_score)
    return max_sentence_score


class AbstractAlgorithm(object):
    def is_follow_up(self, question, history_questions, previous_answer):
        pass

    def train(self, question, history_questions, previous_answer):
        pass

    def save_model(self):
        pass


class DeBoni(AbstractAlgorithm):
    def __init__(self):
        self.q_q_threshold = 0.5
        self.q_a_threshold = 0.5

    def set_q_q_threshold(self, q_q_threshol):
        self.q_q_threshold = q_q_threshol

    def set_q_a_threshold(self, q_a_threshol):
        self.q_a_threshold = q_a_threshol

    def is_follow_up(self, question, history_questions, previous_answer):
        follow_up = False
        if question.has_pronoun() \
                or question.has_cue_word() \
                or not question.has_verb() \
                or sentence_similarity(question, history_questions) > self.q_q_threshold \
                or sentence_similarity(question, previous_answer) > self.q_a_threshold:
            follow_up = True
        return follow_up


class FanYang(AbstractAlgorithm):
    pass


def evaluation(file_name='data/predicted-result.txt'):
    with codecs.open('data/actual-result.txt', encoding='utf-8', mode='rb') as actual:
        with codecs.open(file_name, encoding='utf-8', mode='rb') as predicted:
            result = {'N': {'N': 0, 'F': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0},
                      'F': {'N': 0, 'F': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0}}
            new = result['N']
            follow = result['F']
            for predicted_line in predicted:
                predicted_line = predicted_line.strip()
                actual_line = actual.next()
                actual_line = actual_line.strip()

                result[predicted_line[-1]][actual_line[-1]] += 1

            new['P'] = float(new['N']) / (new['N'] + new['F'])
            new['R'] = float(new['N']) / (new['N'] + follow['N'])
            new['F1'] = 2 * new['P'] * new['R'] / (new['P'] + new['R'])

            follow['P'] = float(follow['F']) / (follow['F'] + follow['N'])
            follow['R'] = float(follow['F']) / (follow['F'] + new['F'])
            follow['F1'] = 2 * follow['P'] * follow['R'] / (follow['P'] + follow['R'])

            return result


def test(algorithm, file_name='data/predicted-result.txt'):
    with codecs.open('data/test-set.txt', encoding='utf-8', mode='rb') as test_set:
        with codecs.open(file_name, encoding='utf-8', mode='wb') as result_file:
            logger.info('start to test all')
            history_questions = []
            previous_answer_text = None
            for line in test_set:
                line = line.strip()
                if line == '':
                    continue
                if line.startswith('A'):
                    previous_answer_text = line.split(':', 1)[1]
                    continue
                [prefix, question_text] = line.split(':', 1)
                question = get_analyzed_result(question_text)
                previous_answer = get_analyzed_result(previous_answer_text)
                result = 'F'
                logger.info('start to test %s', prefix)
                if not algorithm.is_follow_up(question, history_questions, previous_answer):
                    result = 'N'
                    history_questions = []
                logger.info('finished testing %s', prefix)
                result_file.write('{}:{}\n'.format(prefix, result))
                history_questions.append(question)
            logger.info('finished testing all')


def adjust_threshold(q_a_threshold=None, q_q_threshold=None):
    if q_a_threshold is not None and q_q_threshold is not None:
        raise RuntimeError('no more than 1 threshold given but 2')
    de_boni = DeBoni()
    # 调参方式 0-两个都调 1-调q_q 2-调q_a
    scheme = 0
    output_name = 'data/adjust-threshold-both.json'
    if q_a_threshold is not None:
        # q_a_threshold固定
        scheme += 1
        output_name = 'data/adjust-threshold-q-a-{}.json'.format(q_a_threshold)
        logger.info('set constant question-answer similarity threshold=%s',
                    q_a_threshold)
        de_boni.set_q_a_threshold(q_a_threshold)
    if q_q_threshold is not None:
        # q_q_threshold固定
        scheme += 2
        output_name = 'data/adjust-threshold-q-q-{}.json'.format(q_q_threshold)
        logger.info('set constant question-question similarity threshold=%s',
                    q_q_threshold)
        de_boni.set_q_q_threshold(q_q_threshold)
    result = []
    for x in range(0, 11, 1):
        threshold = x / 10.0
        # q_a_threshold固定
        if scheme == 1:
            logger.info('set question-question similarity thresholds=%s',
                        threshold)
            de_boni.set_q_q_threshold(threshold)
            file_name = 'data/q-q-{}-q-a-{}.txt'.format(threshold, q_a_threshold)
        # q_q_threshold固定
        elif scheme == 2:
            logger.info('set question-answer similarity thresholds=%s',
                        threshold)
            de_boni.set_q_a_threshold(threshold)
            file_name = 'data/q-q-{}-q-a-{}.txt'.format(q_q_threshold,
                                                        threshold)
        else:
            logger.info('set all similarity thresholds=%s',
                        threshold)
            de_boni.set_q_a_threshold(threshold)
            de_boni.set_q_q_threshold(threshold)
            file_name = 'data/q-q-{0}-q-a-{0}.txt'.format(threshold)
        if os.path.isfile(file_name):
            logger.info('%s exists', file_name)
        else:
            test(de_boni, file_name=file_name)
        evaluation_result = evaluation(file_name=file_name)
        result.append({'threshold': threshold, 'result': evaluation_result})
    with codecs.open(output_name, encoding='utf-8', mode='wb') as f:
        f.write(json.dumps(result))


def show_usage():
    pass


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'htdea', ['help', 'test-set', 'de-boni', 'evaluation', 'adjust-threshold'])
    except getopt.GetoptError:
        show_usage()
        return
    logging.config.dictConfig(LOGGING)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            show_usage()
            return
        elif opt in ('-t', '--test-set'):
            data_set_generator = DataSetGenerator()
            data_set_generator.generate()
        elif opt in ('-d', '--de-boni'):
            build_word_vector()
            test()
        elif opt in ('-e', '--evaluation'):
            evaluation()
        elif opt in ('-a', '--adjust-threshold'):
            build_word_vector()
            # for x in range(0, 11, 1):
            #     threshold = x / 10.0
            #     adjust_threshold(q_a_threshold=threshold)
            adjust_threshold()


if __name__ == '__main__':
    main(sys.argv[1:])