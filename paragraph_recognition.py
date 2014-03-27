#!/usr/bin/env python
# coding: utf-8
import requests

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
    return AnalyzedResult(response.json())


class AnalyzedResult():
    def __init__(self, json):
        if not isinstance(json, list):
            raise TypeError
        self.json = json

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