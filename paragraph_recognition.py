#!/usr/bin/env python
# coding: utf-8
import codecs
import getopt
import hashlib
import json
import logging
import logging.config
from time import sleep

import requests
import sys

from data_access import (Session,
                         Paragraph,
                         LtpResult, Question)
from log_config import LOG_PROJECT_NAME, LOGGING


LTP_URL = 'http://api.ltp-cloud.com/analysis'
API_KEY = 'u1Q1k8U6tglHca7ZZJ6qTBaq2k0QYwyXNqyE3kVu'
FORMAT = 'json'
PATTERN = 'all'
param = {'api_key': API_KEY,
         'format': FORMAT,
         'pattern': PATTERN,
         'text': None}

Q_Q_THRESHOLD = 1
Q_A_THRESHOLD = 2

THIRD_PERSON_PRONOUN_DICT = dict.fromkeys([u'他', u'她', u'它', u'他们',
                                           u'她们', u'它们',])

DEMONSTRATIVE_PRONOUN_DICT = dict.fromkeys([u'这', u'这儿', u'这么', u'这里',
                                            u'这会儿', u'这样', u'这么样',
                                            u'这些', u'那', u'那儿', u'那么',
                                            u'那里', u'那会儿', u'那样',
                                            u'那么样', u'那些'])

CUE_WORD_DICT = dict.fromkeys([u'所以'])

SYNONYM_DICT = {}
with open('synonym.txt', 'rb') as f:
    for line in f:
        line = line.strip('\n')
        synonym_line = line.split(' ')
        code = synonym_line.pop(0)
        for word in synonym_line:
            if word in SYNONYM_DICT:
                SYNONYM_DICT[word].append(code)
            else:
                SYNONYM_DICT[word] = [code]


with open('stop-word.txt', 'rb') as f:
    STOP_WORD_DICT = dict((line.strip('\n'), True) for line in f)


def is_synonymous(str1, str2):
    for line in SYNONYM_DICT:
        if str1 in line and str2 in line:
            return True
    return False


def build_param(text):
    param['text'] = text
    return param


def analyze(text):
    response = requests.get(LTP_URL, params=build_param(text), timeout=60)
    if not response.ok:
        raise RuntimeError(
            'bad response code %s, %s' % (response.status_code, response.url))
    return response.json()


class AnalyzedResult():
    def __init__(self, result):
        if isinstance(result, unicode):
            self.json = json.loads(result)
        elif isinstance(result, list):
            self.json = json
        else:
            raise TypeError('expecting type is unicode or json, but %s'
                            % result.__class__)

    def has_pronoun(self):
        for pronoun in self.pronoun():
            if pronoun['cont'] in THIRD_PERSON_PRONOUN_DICT \
                    or pronoun['cont'] in DEMONSTRATIVE_PRONOUN_DICT:
                return True

    def has_cue_words(self):
        for word in self.words():
            if word['cont'] in CUE_WORD_DICT:
                return True
        return False

    def pronoun(self):
        yield self.x_pos_tag('r')

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
            if w['cont'] in STOP_WORD_DICT:
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
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_analyzed_result(question_text):
    md5_string = md5(question_text)
    ltp_result = Session.query(LtpResult).filter_by(md5=md5_string).first()
    if ltp_result is not None:
        analyzed_result = AnalyzedResult(ltp_result.json_text)
    else:
        result_json = analyze(question_text)
        save_analyzed_result(md5_string, result_json)
        analyzed_result = AnalyzedResult(result_json)
    return analyzed_result


def save_analyzed_result(md5_string, result_json):
    ltp_result = LtpResult(md5_string,
                           json.dumps(result_json, ensure_ascii=False))
    Session.add(ltp_result)
    try:
        Session.commit()
    except:
        Session.rollback()


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


def calculate_similarity(text, text_list):
    for text_to_compare in text_list:
        # sentence similarity
        score = 0
        for word in text.exclude_stop_words():
            # get word similarity max
            for word_to_compare in text_to_compare.exclude_stop_words():
                if is_synonymous(word['cont'], word_to_compare['cont']):
                    score += 1
                    break
    return 1


def de_boni():
    with open('test-set.txt', 'rb') as test_set:
        with open('predicted-result.txt', 'wb') as result_file:
            history_questions = []
            previous_answer = None
            for line in test_set:
                if line == '':
                    continue
                elif line.startswith('A'):
                    previous_answer = line.split(':', 1)[1]
                    continue
                [prefix, question_text] = line.split(':', 1)
                try:
                    current_question = get_analyzed_result(question_text)
                except:
                    return
                if current_question.has_pronoun() \
                        or current_question.has_cue_words() \
                        or calculate_similarity(current_question,
                                                history_questions) \
                                > Q_Q_THRESHOLD \
                        or calculate_similarity(current_question,
                                                [previous_answer]) \
                                > Q_A_THRESHOLD:
                    follow_up = True
                else:
                    follow_up = False
                    history_questions = []
                result_file.write(
                    '%s:%s\n' % (prefix, 'F' if follow_up else 'N'))
                history_questions.append(current_question)

def show_usage():
    pass


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'ht', ['help', 'test-set'])
    except getopt.GetoptError:
        show_usage()
        return
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            show_usage()
            return
        elif opt in ('-t', '--test-set'):
            generate_test_set()


if __name__ == '__main__':
    main(sys.argv[1:])