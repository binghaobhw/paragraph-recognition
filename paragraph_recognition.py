#!/usr/bin/env python
# coding: utf-8
import getopt
import hashlib
import json
import logging
import logging.config
import math

import requests
import sys

from data_access import (Session,
                         Paragraph,
                         LtpResult, Question)
from log_config import LOG_PROJECT_NAME, LOGGING

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

THIRD_PERSON_PRONOUN_DICT = dict.fromkeys([u'他', u'她', u'它', u'他们',
                                           u'她们', u'它们',])

DEMONSTRATIVE_PRONOUN_DICT = dict.fromkeys([u'这', u'这儿', u'这么', u'这里',
                                            u'这会儿', u'这样', u'这么样',
                                            u'这些', u'那', u'那儿', u'那么',
                                            u'那里', u'那会儿', u'那样',
                                            u'那么样', u'那些'])

CUE_WORD_DICT = dict.fromkeys([u'所以'])


with open('stop-word.txt', 'rb') as f:
    STOP_WORD_DICT = dict((line.strip('\n'), True) for line in f)


def build_word_vector():
    with open('baike-50.vec.txt', 'rb') as f:
        f.next()
        for line in f:
            line = line.strip('\n')
            columns = line.split(' ')
            word = columns[0]
            vector = [float(num_text) for num_text in columns[1:]]
            WORD_VECTOR[word] = vector


def vector_cos(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


def build_param(text):
    param['text'] = text
    return param


def truncate(text):
    split = text.split(' ', 1)
    if len(split) == 1:
        logger.error('no separator')
        return text[:300]
    return split[0]


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


class AnalyzedResult():
    def __init__(self, result):
        if isinstance(result, unicode):
            self.json = json.loads(result)
        elif isinstance(result, list):
            self.json = result
        else:
            raise TypeError('expecting type is unicode or json, but {}'.
                            format(result.__class__))

    def has_pronoun(self):
        result = False
        for pronoun in self.pronoun():
            if pronoun['cont'] in THIRD_PERSON_PRONOUN_DICT \
                    or pronoun['cont'] in DEMONSTRATIVE_PRONOUN_DICT:
                result = True
                break
        logger.info('%s', result)
        return result

    def has_cue_words(self):
        result = False
        for word in self.words():
            if word['cont'] in CUE_WORD_DICT:
                result = True
                break
        logger.info('%s', result)
        return result

    def pronoun(self):
        for r in self.x_pos_tag('r'):
            yield r

    def has_verb(self):
        return self.has_x_pos_tag('v')

    def has_x_pos_tag(self, x):
        for p in self.json:
            for s in p:
                for w in s:
                    if w['pos'] == x:
                        return True
        return False

    def x_pos_tag(self, x):
        for w in self.words():
            if w['pos'] == x:
                yield w

    def sentence(self):
        for p in self.json:
            for s in p:
                yield s

    def words(self):
        for s in self.sentence():
            for w in s:
                yield w

    def exclude_stop_words(self):
        for w in self.words():
            if w['cont'] not in STOP_WORD_DICT:
                yield w


def async_save_analyzed_result():
    logger = logging.getLogger(LOG_PROJECT_NAME + '.async_save')
    subquery = Session.query(Paragraph.question_id.distinct().
                             label('question_id')) \
        .filter_by(is_deleted=0).subquery()
    questions = Session.query(Question) \
        .join(subquery, Question.question_id == subquery.c.question_id)

    for question in questions:
        title = question.title
        try:
            get_analyzed_result(title)
        except:
            logger.error('fail to analyze %s', title, exc_info=True)
            return


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


def generate_test_set():
    question_num = 1
    answer_num = 1
    result_pattern = u'{}{}:{}\n'
    with open('test-set.txt', 'wb') as test_set:
        with open('actual-result.txt', 'wb') as result:
            for paragraph in Session.query(Paragraph).filter(Paragraph.paragraph_id <= 350).filter(Paragraph.is_deleted == 0).all():
                test_lines = [result_pattern.format('Q', question_num, paragraph.question.title)]
                result_lines = [result_pattern.format('Q', question_num, 'N')]
                question_num += 1
                for reply in paragraph.replies:
                    if reply.is_deleted == 1:
                        continue
                    if reply.is_question():
                        test_line = result_pattern.format('Q',  question_num, reply.content)
                        result_line = result_pattern.format('Q',
                                                            question_num,  'F')
                        result_lines.append(result_line)
                        question_num += 1
                    else:
                        test_line = result_pattern.format('A',  answer_num,  reply.content)
                        answer_num += 1
                    test_lines.append(test_line)
                test_lines.append('\n')
                test_set.writelines([s.encode('utf-8') for s in test_lines])
                result.writelines([s.encode('utf-8') for s in result_lines])


def word_similarity(a, b):
    if a in WORD_VECTOR and b in WORD_VECTOR:
        return vector_cos(WORD_VECTOR[a], WORD_VECTOR[b])
    return 0.0


def sentence_similarity(text, text_list):
    # sentence similarity
    max_sentence_score = 0.0
    if text is not None and text_list is not None:
        t_list = text_list if isinstance(text_list, list) else [text_list]
        for text_to_compare in t_list:
            sentence_score = 0.0
            for word in text.exclude_stop_words():
                # word similarity
                max_word_score = 0.0
                for word_to_compare in text_to_compare.exclude_stop_words():
                    word_score = word_similarity(word['cont'], word_to_compare['cont'])
                    logger.info('score=%s, word1=%s, word2=%s', word_score, word['cont'].encode('utf-8'), word_to_compare['cont'].encode('utf-8'))
                    if max_word_score < word_score:
                        max_word_score = word_score
                sentence_score += max_word_score
            if max_sentence_score < sentence_score:
                max_sentence_score = sentence_score
    logger.info('max_sentence_score=%s', max_sentence_score)
    return max_sentence_score


class AbstractAlgorithm():
    def is_follow_up(self, question, history_questions, previous_answer):
        pass

    def train(self, question, history_questions, previous_answer):
        pass

    def save_model(self):
        pass


class DeBoni(AbstractAlgorithm):
    def __init__(self):
        self.q_q_threshold = 1
        self.q_a_threshold = 2

    def is_follow_up(self, question, history_questions, previous_answer):
        follow_up = False
        if question.has_pronoun() \
                or question.has_cue_words() \
                or sentence_similarity(question,
                                        history_questions) \
                        > self.q_q_threshold \
                or sentence_similarity(question,
                                        previous_answer) \
                        > self.q_a_threshold:
            follow_up = True
        return follow_up


class FanYang(AbstractAlgorithm):
    pass


def evaluation():
    with open('actual-result.txt', 'rb') as actual:
        with open('predicted-result.txt', 'rb') as predicted:
            result = {'N': {'N': 0, 'F': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0},
                      'F': {'N': 0, 'F': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0}}
            new = result['N']
            follow = result['F']
            for predicted_line in predicted:
                predicted_line = predicted_line.strip('\n')
                actual_line = actual.next()
                actual_line = actual_line.strip('\n')

                result[predicted_line[-1]][actual_line[-1]] += 1

            new['P'] = float(new['N']) / (new['N'] + new['F'])
            new['R'] = float(new['N']) / (new['N'] + follow['N'])
            new['F1'] = 2 * new['P'] * new['R'] / (new['P'] + new['R'])

            follow['P'] = float(follow['F']) / (follow['F'] + follow['N'])
            follow['R'] = float(follow['F']) / (follow['F'] + new['F'])
            follow['F1'] = 2 * follow['P'] * follow['R'] / (follow['P'] + follow['R'])

            print json.dumps(result)


def test():
    de_boni = DeBoni()
    with open('test-set.txt', 'rb') as test_set:
        with open('predicted-result.txt', 'wb') as result_file:
            history_questions = []
            previous_answer_text = None
            for line in test_set:
                line = line.strip('\n')
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
                if not de_boni.is_follow_up(question, history_questions, previous_answer):
                    result = 'N'
                    history_questions = []
                logger.info('finished testing %s', prefix)
                result_file.write('{}:{}\n'.format(prefix, result))
                history_questions.append(question)


def show_usage():
    pass


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'htde', ['help', 'test-set', 'de-boni', 'evaluation'])
    except getopt.GetoptError:
        show_usage()
        return
    logging.config.dictConfig(LOGGING)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            show_usage()
            return
        elif opt in ('-t', '--test-set'):
            generate_test_set()
        elif opt in ('-d', '--de-boni'):
            test()
        elif opt in ('-e', '--evaluation'):
            evaluation()


if __name__ == '__main__':
    main(sys.argv[1:])