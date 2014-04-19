#!/usr/bin/env python
# coding: utf-8
import codecs
from collections import deque
import getopt
import hashlib
import json
import logging
import logging.config
import os
import sys
import re
import requests
from data_access import Session, LtpResult, Paragraph
from log_config import LOG_PROJECT_NAME, LOGGING
from method import AnalyzedSentence
import method

logger = logging.getLogger(LOG_PROJECT_NAME + '.experiment')
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


def md5(text):
    data = text.encode('utf-8') if isinstance(text, unicode) else text
    return hashlib.md5(data).hexdigest()


def truncate(text):
    for punctuation in PUNCTUATION_TABLE:
        index = text.find(punctuation)
        if 0 < index < 50:
            return text[:index]
    return text[:50]


def analyze(text):
    response = requests.get(LTP_URL, params=build_param(text), timeout=60)
    if not response.ok:
        if response.status_code == 400 and \
                response.json()['error_message'] == 'SENTENCE TOO LONG':
            logger.info('sentence too long, truncate')
            truncated_text = truncate(text)
            response = requests.get(LTP_URL,
                                    params=build_param(truncated_text),
                                    timeout=60)
        else:
            raise RuntimeError('bad response code={} url={} text={}'.format(
                response.status_code, response.url, response.text))
    return response.json()


def save_analyzed_result(md5_string, result_json):
    ltp_result = LtpResult(md5_string,
                           json.dumps(result_json, ensure_ascii=False))
    Session.add(ltp_result)
    logger.info('start to insert ltp result, md5=%s', md5_string)
    try:
        Session.commit()
    except Exception:
        Session.rollback()
        logger.error('fail to insert', exc_info=True)
    logger.info('finished inserting ltp result')


def get_analyzed_result(question_text):
    if question_text is None:
        return None
    md5_string = md5(question_text)
    ltp_result = Session.query(LtpResult).filter_by(md5=md5_string).first()
    if ltp_result is not None:
        analyzed_result = AnalyzedSentence(md5_string, ltp_result.json_text)
    else:
        try:
            result_json = analyze(question_text)
        except RuntimeError:
            logger.error('fail to invoke ltp api, text=%s', question_text,
                         exc_info=True)
            raise RuntimeError()

        save_analyzed_result(md5_string, result_json)
        analyzed_result = AnalyzedSentence(md5_string, result_json)
    return analyzed_result


def test(method_, file_name='data/predicted-result.txt'):
    with codecs.open('data/test-set.txt', encoding='utf-8') as test_set, \
            codecs.open(file_name, encoding='utf-8', mode='wb') as result_file:
        logger.info('start to test all')
        history_questions = []
        previous_answer_text = None
        last_is_answer = False
        for line in test_set:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('A'):
                previous_answer_text = line.split(':', 1)[1]
                last_is_answer = True
                continue
            [prefix, question_text] = line.split(':', 1)
            question = get_analyzed_result(question_text)
            previous_answer = get_analyzed_result(previous_answer_text) \
                if last_is_answer else None
            last_is_answer = False
            result = 'F'
            logger.info('start to test %s', prefix)
            if not method_.is_follow_up(question, history_questions,
                                       previous_answer):
                result = 'N'
                history_questions = []
            logger.info('finished testing %s', prefix)
            result_file.write('{}:{}\n'.format(prefix, result))
            history_questions.append(question)
        logger.info('finished testing all')


def evaluation(file_name):
    answer_file_name = 'data/actual-result.txt'
    with codecs.open(answer_file_name, encoding='utf-8') as actual, \
            codecs.open(file_name, encoding='utf-8') as predicted:
        result = {'N': {'N': 0, 'F': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0},
                  'F': {'N': 0, 'F': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0},
                  'A': {'T': 0, 'F': 0, 'P': 0.0}}
        new = result['N']
        follow = result['F']
        all = result['A']
        for predicted_line in predicted:
            predicted_line = predicted_line.strip()
            actual_line = actual.next()
            actual_line = actual_line.strip()
            result[predicted_line[-1]][actual_line[-1]] += 1
        new['P'] = ratio(new['N'], new['F'])
        new['R'] = ratio(new['N'], follow['N'])
        new['F1'] = 2*new['R']*ratio(new['P'], new['R'])
        follow['P'] = ratio(follow['F'], follow['N'])
        follow['R'] = ratio(follow['F'], new['F'])
        follow['F1'] = 2*follow['R']*ratio(follow['P'], follow['R'])
        all['T'] = new['N'] + follow['F']
        all['F'] = new['F'] + follow['N']
        all['P'] = ratio(all['T'], all['F'])
    return result


def ratio(a, b):
    total = a+b
    result = 0.0
    if total != 0:
        result = float(a) / total
    return result


def adjust_threshold(path, q_a_threshold=None, q_q_threshold=None):
    if q_a_threshold is not None and q_q_threshold is not None:
        raise RuntimeError('no more than 1 threshold given but 2')
    method_ = method.get_method('de_boni')
    # 调参方式 0-两个都调 1-调q_q 2-调q_a
    scheme = 0
    output_name = '{}/adjust-threshold-both.json'.format(path)
    if q_a_threshold is not None:
        # q_a_threshold固定
        scheme += 1
        output_name = '{}/adjust-threshold-q-a-{}.json'.format(path,
                                                               q_a_threshold)
        logger.info('set constant question-answer similarity threshold=%s',
                    q_a_threshold)
        method_.q_a_threshold = q_a_threshold
    if q_q_threshold is not None:
        # q_q_threshold固定
        scheme += 2
        output_name = '{}/adjust-threshold-q-q-{}.json'.format(path,
                                                               q_q_threshold)
        logger.info('set constant question-question similarity threshold=%s',
                    q_q_threshold)
        method_.q_q_threshold = q_q_threshold
    result = []
    for x in range(80, 100, 1):
        threshold = x / 100.0
        # q_a_threshold固定
        if scheme == 1:
            logger.info('set question-question similarity thresholds=%s',
                        threshold)
            method_.q_q_threshold = threshold
            file_name = '{}/q-q-{}-q-a-{}.txt'.format(path, threshold,
                                                      q_a_threshold)
        # q_q_threshold固定
        elif scheme == 2:
            logger.info('set question-answer similarity thresholds=%s',
                        threshold)
            method_.q_a_threshold = threshold
            file_name = '{}/q-q-{}-q-a-{}.txt'.format(path, q_q_threshold,
                                                      threshold)
        else:
            logger.info('set all similarity thresholds=%s',
                        threshold)
            method_.q_a_threshold = threshold
            method_.q_q_threshold = threshold
            file_name = '{0}/q-q-{1}-q-a-{1}.txt'.format(path, threshold)
        if os.path.isfile(file_name):
            logger.info('%s exists', file_name)
        else:
            test(method_, file_name=file_name)
        evaluation_result = evaluation(file_name=file_name)
        result.append({'threshold': threshold, 'result': evaluation_result})
    with codecs.open(output_name, encoding='utf-8', mode='wb') as f:
        f.write(json.dumps(result))


class DataSetGenerator():
    def __init__(self, data_set_file_name='data/test-set.txt',
                 answer_file_name='data/true-result.txt'):
        self.result_pattern = u'{}{}:{}\n'
        self.queue = deque()
        self.question_num = 1
        self.answer_num = 1
        self.data_set_file_name = data_set_file_name
        self.answer_file_name = answer_file_name

    def generate_paragraph(self, paragraph):
        paragraph_lines = [self.result_pattern.format(
            'Q', self.question_num,  paragraph.question.title)]
        result_lines = [self.result_pattern.format('Q', self.question_num,
                                                   'N')]
        self.question_num += 1
        for reply in paragraph.replies:
            if reply.is_deleted == 1:
                continue
            if reply.is_question():
                test_line = self.result_pattern.format('Q',  self.question_num,
                                                       reply.content)
                result_line = self.result_pattern.format(
                    'Q', self.question_num, 'F')
                result_lines.append(result_line)
                self.question_num += 1
            else:
                test_line = self.result_pattern.format('A', self.answer_num,
                                                       reply.content)
                self.answer_num += 1
            paragraph_lines.append(test_line)
        return paragraph_lines, result_lines

    def generate(self):
        previous_category_id = None
        with codecs.open(self.data_set_file_name, encoding='utf-8',
                         mode='wb') as data_set, codecs.open(
                self.answer_file_name, encoding='utf-8', mode='wb') as result:
            logger.info('start to generate data set')
            for paragraph in Session.query(Paragraph).filter(
                    Paragraph.paragraph_id <= 600).filter(
                        Paragraph.is_deleted == 0).all():
                # 当前分类与上一话段分类一样，先不写，入队列
                if paragraph.question.category_id == previous_category_id:
                    logger.debug('same category id, put %s into queue',
                                paragraph.paragraph_id)
                    self.queue.append(paragraph)
                    continue
                paragraph_lines, result_lines = self.generate_paragraph(
                    paragraph)
                paragraph_lines.append('\n')
                data_set.writelines(paragraph_lines)
                result.writelines(result_lines)
                previous_category_id = paragraph.question.category_id
                # 看队列头是否能写
                if len(self.queue) > 0:
                    paragraph_in_queue = self.queue.popleft()
                    if paragraph_in_queue.question.category_id == \
                            previous_category_id:
                        logger.debug('same category id again, put %s into '
                                    'queue', paragraph_in_queue.paragraph_id)
                        self.queue.append(paragraph_in_queue)
                    else:
                        logger.debug('dequeue')
                        paragraph_lines, result_lines = self.\
                            generate_paragraph(paragraph_in_queue)
                        paragraph_lines.append('\n')
                        data_set.writelines(paragraph_lines)
                        result.writelines(result_lines)
                        previous_category_id = paragraph_in_queue.question.\
                            category_id
            for paragraph_in_queue in self.queue:
                paragraph_lines, result_lines = self.generate_paragraph(
                    paragraph_in_queue)
                paragraph_lines.append('\n')
                data_set.writelines(paragraph_lines)
                result.writelines(result_lines)
            logger.info('finished generating data set')


def train_data(method_, train_set_file_name, train_data_file_name):
    with codecs.open(train_set_file_name, encoding='utf-8') as train_set, \
            codecs.open('data/true-result.txt', encoding='utf-8') as \
                    result_file, codecs.open(train_data_file_name,
                                             encoding='utf-8', mode='wb') as \
            output:
        context_window = 5
        history_questions = deque(maxlen=context_window)
        previous_answer_text = None
        last_is_answer = False
        output.write('"","result","pronoun","proper_noun","noun","verb",'
                     '"max_sentence_similarity"\n')
        for line in train_set:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('A'):
                previous_answer_text = line.split(':', 1)[1]
                last_is_answer = True
                continue
            result = result_file.next().strip().split(':', 1)[1]
            [prefix, question_text] = line.split(':', 1)
            num = prefix[1:]
            question = get_analyzed_result(question_text)
            previous_answer = get_analyzed_result(previous_answer_text) if \
                last_is_answer else None
            features = method_.features(question, history_questions,
                                        previous_answer)
            output.write('{0},{1},{2[0]},{2[1]},{2[2]},{2[3]},{2[4]}\n'.
                         format(num, result, features))
            history_questions.append(question)
            last_is_answer = False


def adjust_len(path, output):
    file_list = os.listdir(path)
    len_dict = dict([(int(re.findall(r'\d+', f)[0]), '{}/{}'.format(path, f))
                     for f in file_list])
    result = []
    for len in sorted(len_dict.keys()):
        evaluation_result = evaluation(file_name=len_dict[len])
        result.append({'threshold': len, 'result': evaluation_result})
    with codecs.open(output, encoding='utf-8', mode='wb') as f:
        f.write(json.dumps(result))


def show_usage():
    pass


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'hgtea', ['help', 'generate', 'test',
                                                   'evaluation',
                                                   'adjust-threshold',
                                                   'train-data'])
    except getopt.GetoptError:
        show_usage()
        return
    logging.config.dictConfig(LOGGING)
    method_config = {
        'essentials': {
            'third_person_pronoun': 'data/third-person-pronoun.txt',
            'demonstrative_pronoun': 'data/demonstrative-pronoun.txt',
            'cue_word': 'data/cue-word.txt',
            'stop_word': 'data/stop-word.txt'
        },
        'word_similarity_calculators': {
            'word_embedding': {
                'class': 'WordEmbeddingCalculator',
                'vector_file_name': 'data/baike-50.vec.txt'
            }
        },
        'sentence_similarity_calculator': {
            'ssc': {
                'cache': True,
                'cache_file_name': 'data/sentence-score-cache',
                'word_similarity_calculator': 'word_embedding'
            }
        },
        'method': {
            'de_boni': {
                'class': 'DeBoni',
                'sentence_similarity_calculator': 'ssc'
            },
            'fan_yang': {
                'class': 'FanYang',
                'sentence_similarity_calculator': 'ssc'
            }
        }
    }
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            show_usage()
            return
        elif opt in ('-g', '--generate'):
            data_set_generator = DataSetGenerator()
            data_set_generator.generate()
        elif opt in ('-t', '--test'):
            method.configure(method_config)
            method_ = method.get_method(arg)
            test(method_, 'data/q-q-0.89-q-a-0.89.txt')
        elif opt in ('-e', '--evaluation'):
            print evaluation('data/{}'.format('q-q-1.0-q-a-1.0.txt'))
        elif opt in ('-a', '--adjust-threshold'):
            method.configure(method_config)
            adjust_threshold('data/pid-500-len-10-pcvs')
        elif opt in '--train-data':
            method.configure(method_config)
            method_ = method.get_method('fan_yang')
            train_data(method_, 'data/test-set.txt', 'data/train-data.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
