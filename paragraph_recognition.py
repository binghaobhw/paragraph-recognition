#!/usr/bin/env python
# coding: utf-8
import hashlib
import json
import logging
import logging.config
from time import sleep

import requests

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


def build_param(text):
    param['text'] = text
    return param


def analyze(text):
    response = requests.get(LTP_URL, params=build_param(text), timeout=60)
    if not response.ok:
        raise RuntimeError('bad response code %s, %s' % (response.status_code, response.url))
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

    def has_third_person_pronoun(self):
        return True

    def has_cue_words(self):
        return True
    def has_pronoun(self):
        return self.has_x_pos_tag('r')

    def has_verb(self):
        return self.has_x_pos_tag('v')

    def has_x_pos_tag(self, x):
        for p in self.json:
            for s in p:
                for w in s:
                    if w['pos'] == x:
                        return True
        return False


def async_save_analyzed_result():
    logger = logging.getLogger(LOG_PROJECT_NAME + '.async_save')
    subquery = Session.query(Paragraph.question_id.distinct().
                             label('question_id'))\
        .filter_by(is_deleted=0).subquery()
    questions = Session.query(Question)\
        .join(subquery, Question.question_id == subquery.c.question_id)

    for question in questions:
        title = question.title
        try:
            r = get_analyzed_result(title)
        except:
            logger.error('fail to analyze %s', title, exc_info=True)
            return


def md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_analyzed_result(question_text):
    md5_string = md5(question_text)
    ltp_result = Session.query(LtpResult).filter_by(md5=md5_string).first()
    if ltp_result is not None:
        result_json = json.loads(ltp_result.analyzed_result)
    else:
        result_json = analyze(question_text)
        save_analyzed_result(md5_string, result_json)
    return result_json


def save_analyzed_result(md5_string, result_json):
    ltp_result = LtpResult(md5_string, json.dumps(result_json, ensure_ascii=False))
    Session.add(ltp_result)
    try:
        Session.commit()
    except:
        Session.rollback()


def generate_test_set():
    pass

def calculate_similarity(text, text_list):
    return 1


Q_Q_THRESHOLD = 1
Q_A_THRESHOLD = 2

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
                    pass
                if current_question.has_third_person_pronoun() or current_question.has_cue_words() or calculate_similarity(current_question, history_questions) > Q_Q_THRESHOLD or calculate_similarity(current_question, [previous_answer]) > Q_A_THRESHOLD:
                    follow_up = True
                else:
                    follow_up = False
                    history_questions = []
                result_file.write('%s:%s\n' % (prefix, 'F' if follow_up else 'N'))
                history_questions.append(current_question)










def main():
    logging.config.dictConfig(LOGGING)
    async_save_analyzed_result()
    # r = Session.query(LtpResult).all()
    # for ltp_result in r:
    #     a = AnalyzedResult(ltp_result.analyzed_result)
    #     pass
    #
    # q = Session.query(Question).filter_by(question_id=807911036005241572).first()
    # question = q.title
    # ltp_result = Session.query(LtpResult).filter_by(md5=hashlib.md5(question
    #                                                             .encode('utf-8'))
    #                            .hexdigest()).first()
    # a = AnalyzedResult(ltp_result.analyzed_result)
    # a.has_verb()


if __name__ == '__main__':
    main()