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
    def __init__(self, json_string):
        if not isinstance(json_string, unicode):
            raise TypeError('expecting type is unicode but %s' % json_string
                            .__class__)
        self.json = json.loads(json_string)

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
        md5_string = md5(title)
        r = get_analyzed_result(md5_string)
        if r is None:
            try:
                analyzed_result = analyze(title)
            except:
                logger.error('fail to analyze %s', title, exc_info=True)
                return
            logger.info('start to insert %s', md5_string)
            try:
                save_analyzed_result(md5_string, analyzed_result)
            except:
                logger.error('fail to insert %s', title, exc_info=True)
                return
            logger.info('finished inserting %s', md5_string)
            sleep(1)


def md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_analyzed_result(md5):
    return Session.query(LtpResult).filter_by(md5=md5).first()


def save_analyzed_result(md5_string, analyzed_result):
    ltp_result = LtpResult(md5_string, json.dumps(analyzed_result, ensure_ascii=False))
    Session.add(ltp_result)
    Session.commit()


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