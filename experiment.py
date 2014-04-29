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


def test(method_, test_set_filename, result_filename):
    """Use method_ to judge questions in test_set_filename.

    :param method_: subclass of AbstractMethod
    :param test_set_filename: str
    :param result_filename: str
    """
    with codecs.open(test_set_filename, encoding='utf-8') as test_set, \
            codecs.open(result_filename, encoding='utf-8', mode='wb') as result_file:
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
            logger.info('start to test %s', prefix)
            follow_up = method_.is_follow_up(question, history_questions,
                                             previous_answer)
            logger.info('finished testing %s, follow_up: %s', prefix,
                        follow_up)
            if not follow_up:
                history_questions = []
            result_file.write('{}:{:d}\n'.format(prefix, follow_up))
            history_questions.append(question)
        logger.info('finished testing all')


def evaluation(result_filename, label_filename):
    """Evaluate the test result.

    :param result_filename: str
    :param label_filename: str
    :return: dict
    """
    with codecs.open(label_filename, encoding='utf-8') as label_file, \
            codecs.open(result_filename, encoding='utf-8') as result_file:
        outcome = {
            'new': {'new': 0, 'follow': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0},
            'follow': {'new': 0, 'follow': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0},
            'all': {'true': 0, 'false': 0, 'P': 0.0}}
        num_meaning = {'1': 'new', '0': 'follow'}
        new = outcome['new']
        follow = outcome['follow']
        all_ = outcome['all']
        for result_line in result_file:
            result_line = result_line.strip()
            label_line = label_file.next()
            label_line = label_line.strip()
            result_key = num_meaning[result_line[-1]]
            label_key = num_meaning[label_line[-1]]
            outcome[result_key][label_key] += 1
        new['P'] = ratio(new['new'], new['follow'])
        new['R'] = ratio(new['new'], follow['new'])
        new['F1'] = 2*new['R']*ratio(new['P'], new['R'])
        follow['P'] = ratio(follow['follow'], follow['new'])
        follow['R'] = ratio(follow['follow'], new['follow'])
        follow['F1'] = 2*follow['R']*ratio(follow['P'], follow['R'])
        all_['true'] = new['new'] + follow['follow']
        all_['false'] = new['follow'] + follow['new']
        all_['P'] = ratio(all_['true'], all_['false'])
    return outcome


def ratio(a, b):
    """Return a/(a+b)

    :param a: int | float
    :param b: int | float
    :return: float
    """
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
            test(method_, result_filename=file_name)
        evaluation_result = evaluation(result_filename=file_name)
        result.append({'threshold': threshold, 'result': evaluation_result})
    with codecs.open(output_name, encoding='utf-8', mode='wb') as f:
        f.write(json.dumps(result))


class DatasetGenerator(object):
    def __init__(self, dataset_filename='data/test-set.txt',
                 label_filename='data/label.txt'):
        self.result_pattern = u'{}{}:{}\n'
        self.queue = deque()
        self.question_num = 1
        self.answer_num = 1
        self.dataset_filename = dataset_filename
        self.label_filename = label_filename

    def generate_paragraph(self, paragraph):
        paragraph_lines = [self.result_pattern.format(
            'Q', self.question_num,  paragraph.question.title)]
        label_lines = [self.result_pattern.format('Q', self.question_num, 0)]
        self.question_num += 1
        for reply in paragraph.replies:
            if reply.is_deleted == 1:
                continue
            if reply.is_question():
                test_line = self.result_pattern.format('Q',  self.question_num,
                                                       reply.content)
                label_line = self.result_pattern.format('Q',
                                                         self.question_num, 1)
                label_lines.append(label_line)
                self.question_num += 1
            else:
                test_line = self.result_pattern.format('A', self.answer_num,
                                                       reply.content)
                self.answer_num += 1
            paragraph_lines.append(test_line)
        return paragraph_lines, label_lines

    def generate(self, num):
        """generate dataset and label.

        dataset format:
            Q1:question1
            A1:answer1
            Q2:question2

            Q3:question3
            A2:answer2

        label format: 0: new 1: follow-up
            Q1:0
            Q2:1
            Q3:0

        :param num: int, number of paragraphs
        """
        previous_category_id = None
        with codecs.open(self.dataset_filename, encoding='utf-8', mode='wb') \
                as dataset_file, codecs.open(self.label_filename,
                                             encoding='utf-8', mode='wb') as \
                label_file:
            logger.info('start to generate dataset, limit %s', num)
            for paragraph in Session.query(Paragraph).filter(
                    Paragraph.is_deleted == 0).order_by(
                    Paragraph.paragraph_id).limit(num):
                # When current question has same category with the previous,
                # put it into queue.
                if paragraph.question.category_id == previous_category_id:
                    logger.debug('same category id, put %s into queue',
                                paragraph.paragraph_id)
                    self.queue.append(paragraph)
                    continue
                paragraph_lines, label_lines = self.generate_paragraph(
                    paragraph)
                paragraph_lines.append('\n')
                dataset_file.writelines(paragraph_lines)
                label_file.writelines(label_lines)
                previous_category_id = paragraph.question.category_id
                # Output the head of queue when its category is different with
                # the previous, otherwise put into queue again.
                if len(self.queue) > 0:
                    paragraph_in_queue = self.queue.popleft()
                    if paragraph_in_queue.question.category_id == \
                            previous_category_id:
                        logger.debug('same category id again, put %s into '
                                     'queue', paragraph_in_queue.paragraph_id)
                        self.queue.append(paragraph_in_queue)
                    else:
                        logger.debug('dequeue')
                        paragraph_lines, label_lines = self.\
                            generate_paragraph(paragraph_in_queue)
                        paragraph_lines.append('\n')
                        dataset_file.writelines(paragraph_lines)
                        label_file.writelines(label_lines)
                        previous_category_id = paragraph_in_queue.question.\
                            category_id
            for paragraph_in_queue in self.queue:
                paragraph_lines, label_lines = self.generate_paragraph(
                    paragraph_in_queue)
                paragraph_lines.append('\n')
                dataset_file.writelines(paragraph_lines)
                label_file.writelines(label_lines)
            logger.info('finished generating data set')


def train_data(method_, dataset_file_name, label_file_name,
               train_set_file_name):
    with codecs.open(dataset_file_name, encoding='utf-8') as dataset_file, \
            codecs.open(label_file_name, encoding='utf-8') as label_file, \
            codecs.open(train_set_file_name, encoding='utf-8', mode='wb') as \
            train_set_file:
        context_window = 5
        history_questions = deque(maxlen=context_window)
        previous_answer_text = None
        last_is_answer = False
        feature_names = method_.feature_names
        head = ','.join(feature_names)
        head = '{},label\n'.format(head)
        train_set_file.write(head)
        for line in dataset_file:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('A'):
                previous_answer_text = line.split(':', 1)[1]
                last_is_answer = True
                continue
            label = label_file.next().strip().split(':', 1)[1]
            question_text = line.split(':', 1)[1]
            question = get_analyzed_result(question_text)
            previous_answer = get_analyzed_result(previous_answer_text) if \
                last_is_answer else None
            features = method_.features(question, history_questions,
                                        previous_answer)
            train_data_line = to_literal(features + [label])
            train_data_line = ','.join(train_data_line)
            train_set_file.write('{}\n'.format(train_data_line))
            history_questions.append(question)
            last_is_answer = False


def to_literal(x):
    """Convert items in x to literal string.

    :param x: list[unicode | bool | float | int]
    :return: list[unicode]
    :raise RuntimeError:
    """
    result = []
    for i in x:
        if isinstance(i, unicode):
            result.append(i)
        elif isinstance(i, bool):
            result.append(u'1' if i else u'0')
        elif isinstance(i, int):
            result.append(unicode(i))
        elif isinstance(i, float):
            result.append('{:.3f}'.format(i))
        else:
            raise RuntimeError(u'cannot handle {}'.format(i))
    return result


def adjust_len(path, output):
    file_list = os.listdir(path)
    len_dict = dict([(int(re.findall(r'\d+', f)[0]), '{}/{}'.format(path, f))
                     for f in file_list])
    result = []
    for len in sorted(len_dict.keys()):
        evaluation_result = evaluation(result_filename=len_dict[len])
        result.append({'threshold': len, 'result': evaluation_result})
    with codecs.open(output, encoding='utf-8', mode='wb') as f:
        f.write(json.dumps(result))


method_config = {
    'essentials': {
        'third_person_pronoun': 'data/third-person-pronoun.txt',
        'demonstrative_pronoun': 'data/demonstrative-pronoun.txt',
        'cue_word': 'data/cue-word.txt',
        'stop_word': 'data/stop-word.txt'
    },
    'word_similarity_calculators': {
        'how_net': {
            'class': 'HowNetCalculator',
            'sememe_tree_file': 'data/whole.dat',
            'glossary_file': 'data/glossary.dat'
        }
    },
    'sentence_similarity_calculator': {
        'ssc_with_how_net': {
            'word_similarity_calculator': 'how_net',
            'score_filename': 'data/how-net-sentence.score'
        }
    },
    'method': {
        'fan_yang': {
            'class': 'FanYang',
            'sentence_similarity_calculator': 'ssc_with_how_net',
            'train_data_filename': 'data/train-set.txt',
            'classifier_filename': 'data/fan-yang.classifier'
        },
        'de_boni': {
            'class': 'DeBoni',
            'sentence_similarity_calculator': 'ssc_with_how_net',
            'q_q_threshold': 0.89,
            'q_a_threshold': 0.89
        }
    }
}


def prepare():
    logging.config.dictConfig(LOGGING)


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
                'sentence_similarity_calculator': 'ssc',
                'train_data_file': 'data/train-data.csv'
            }
        }
    }
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            show_usage()
            return
        elif opt in ('-g', '--generate'):
            data_set_generator = DatasetGenerator()
            data_set_generator.generate(0)
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
    method.configure(method_config)
    fan_yang = method.get_method('fan_yang')
    train_data(fan_yang, 'data/train-set-data.txt', 'data/train-set-label.txt',
               'data/train-data.txt')
