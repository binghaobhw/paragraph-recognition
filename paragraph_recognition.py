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


def parse(text):
    response = requests.get(LTP_URL, params=build_param(text))
    return response

