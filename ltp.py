#!/usr/bin/env python
# coding: utf-8
import logging

import requests

from log_config import LOG_PROJECT_NAME


logger = logging.getLogger(LOG_PROJECT_NAME + '.ltp')
logger.addHandler(logging.NullHandler())

LTP_URL = 'http://api.ltp-cloud.com/analysis'
API_KEY = 'u1Q1k8U6tglHca7ZZJ6qTBaq2k0QYwyXNqyE3kVu'
FORMAT = 'json'
PATTERN = 'all'
param = {'api_key': API_KEY,
         'format': FORMAT,
         'pattern': PATTERN,
         'text': None}
PUNCTUATION_TABLE = [u' ', u'.', u'。', u',', u'，', u'!', u'！', u';', u'；',
                     u'﹖', u'?', u'？', u'～', u'~']


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
    """Return LTP analyzed result.

    :param text: unicode
    :return: json :raise RuntimeError:
    """
    logger.debug('Invoke ltp api, %s', text)
    response = requests.get(LTP_URL, params=build_param(text), timeout=60)
    if (response.status_code == 400 and
            response.json()['error_message'] == 'SENTENCE TOO LONG') or \
            (response.ok and response.text.startswith('<html')):
        logger.debug('Sentence too long, truncate')
        truncated_text = truncate(text)
        response = requests.get(LTP_URL,
                                params=build_param(truncated_text),
                                timeout=60)
    if response.ok:
        return response.json()
    else:
        raise RuntimeError('bad response code={} url={} text={}'.format(
            response.status_code, response.url, response.text))