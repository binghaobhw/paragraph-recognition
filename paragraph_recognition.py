#!/usr/bin/env python
# coding: utf-8
import hashlib
import json

import requests

from data_access import (Session,
                         Paragraph,
                         LtpResult)


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
    response = requests.get(LTP_URL, params=build_param(text))
    if not response.ok:
        return None
    return response.json()


class AnalyzedResult():
    def __init__(self, json_string):
        if not isinstance(json_string, list):
            raise TypeError
        self.json_string = json_string

    def has_pronoun(self):
        return self.has_x_pos_tag('r')

    def has_verb(self):
        return self.has_x_pos_tag('v')

    def has_x_pos_tag(self, x):
        for p in self.json_string:
            for s in p:
                for w in s:
                    if w['pos'] == x:
                        return True
        return False


def save_analyzed_result():
    r = Session.query(Paragraph).filter(
        Paragraph.is_deleted == 0).order_by(Paragraph.paragraph_id).limit(10) \
        .all()
    for p in r:
        question = p.question.title
        analyzed_result = analyze(question)
        if analyzed_result is None:
            return
        ltp_result = LtpResult(hashlib.md5(question.encode('utf-8'))
                               .hexdigest(),  json.dumps(analyzed_result))
        Session.add(ltp_result)

    Session.commit()


def main():
    save_analyzed_result()


if __name__ == '__main__':
    main()