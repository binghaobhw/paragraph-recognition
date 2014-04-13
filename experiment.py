#!/usr/bin/env python
# coding: utf-8
from Queue import Queue
import codecs
import getopt
import hashlib
import json
import logging
import os
import sys
import requests
from data_access import Session, LtpResult, Paragraph
from log_config import LOG_PROJECT_NAME, LOGGING
from paragraph_recognition import AnalyzedSentence
import paragraph_recognition

logger = logging.getLogger(LOG_PROJECT_NAME + '.paragraph_recognition')
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
        if response.status_code == 400 and response.json()['error_message'] == 'SENTENCE TOO LONG':
            logger.info('sentence too long, truncate')
            truncated_text = truncate(text)
            response = requests.get(LTP_URL, params=build_param(truncated_text), timeout=60)
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
    except:
        Session.rollback()
        logger.error('fail to insert')
    logger.info('finished inserting ltp result')


def get_analyzed_result(question_text):
    if question_text is None:
        return None
    md5_string = md5(question_text)
    ltp_result = Session.query(LtpResult).filter_by(md5=md5_string).first()
    if ltp_result is not None:
        analyzed_result = AnalyzedSentence(ltp_result.json_text)
    else:
        try:
            result_json = analyze(question_text)
        except RuntimeError:
            logger.error('fail to invoke ltp api, text=%s', question_text, exc_info=True)
            raise RuntimeError()

        save_analyzed_result(md5_string, result_json)
        analyzed_result = AnalyzedSentence(result_json)
    return analyzed_result


def build_essentials(path):
    file_dict = {
        'third_person_pronoun_dict': path + '/third-person-pronoun.txt',
        'demonstrative_pronoun_dict': path + '/demonstrative-pronoun-dict.txt',
        'cue_word_dict': path + '/cue-word.txt',
        'stop_word_dict': path + '/stop-word.txt'
    }
    essentials = {}
    for name, path in file_dict.items():
        with codecs.open(path, encoding='utf-8', mode='rb') as f:
            essentials[name] = dict.fromkeys([line.strip() for line in f])
    return essentials


def build_word_embedding_vectors():
    with codecs.open('data/baike-50.vec.txt', encoding='utf-8', mode='rb') as f:
        word_embedding_vectors = {}
        logger.info('start to build word vector')
        f.next()
        for line in f:
            line = line.strip()
            columns = line.split(' ')
            word = columns[0]
            vector = [float(num_text) for num_text in columns[1:]]
            word_embedding_vectors[word] = vector
        logger.info('finished building word vector')
        return word_embedding_vectors


def test(method, file_name='data/predicted-result.txt'):
    with codecs.open('data/test-set.txt', encoding='utf-8', mode='rb') as test_set:
        with codecs.open(file_name, encoding='utf-8', mode='wb') as result_file:
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
                if not method.is_follow_up(question, history_questions, previous_answer):
                    result = 'N'
                    history_questions = []
                logger.info('finished testing %s', prefix)
                result_file.write('{}:{}\n'.format(prefix, result))
                history_questions.append(question)
            logger.info('finished testing all')


def evaluation(file_name='data/predicted-result.txt'):
    with codecs.open('data/actual-result.txt', encoding='utf-8', mode='rb') as actual:
        with codecs.open(file_name, encoding='utf-8', mode='rb') as predicted:
            result = {'N': {'N': 0, 'F': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0},
                      'F': {'N': 0, 'F': 0, 'P': 0.0, 'R': 0.0, 'F1': 0.0}}
            new = result['N']
            follow = result['F']
            for predicted_line in predicted:
                predicted_line = predicted_line.strip()
                actual_line = actual.next()
                actual_line = actual_line.strip()

                result[predicted_line[-1]][actual_line[-1]] += 1

            new['P'] = float(new['N']) / (new['N'] + new['F'])
            new['R'] = float(new['N']) / (new['N'] + follow['N'])
            new['F1'] = 2 * new['P'] * new['R'] / (new['P'] + new['R'])

            follow['P'] = float(follow['F']) / (follow['F'] + follow['N'])
            follow['R'] = float(follow['F']) / (follow['F'] + new['F'])
            follow['F1'] = 2 * follow['P'] * follow['R'] / (follow['P'] + follow['R'])

            return result


def adjust_threshold(q_a_threshold=None, q_q_threshold=None):
    if q_a_threshold is not None and q_q_threshold is not None:
        raise RuntimeError('no more than 1 threshold given but 2')
    method = paragraph_recognition.get_method('de_boni')
    # 调参方式 0-两个都调 1-调q_q 2-调q_a
    scheme = 0
    output_name = 'data/adjust-threshold-both.json'
    if q_a_threshold is not None:
        # q_a_threshold固定
        scheme += 1
        output_name = 'data/adjust-threshold-q-a-{}.json'.format(q_a_threshold)
        logger.info('set constant question-answer similarity threshold=%s',
                    q_a_threshold)
        method.set_q_a_threshold(q_a_threshold)
    if q_q_threshold is not None:
        # q_q_threshold固定
        scheme += 2
        output_name = 'data/adjust-threshold-q-q-{}.json'.format(q_q_threshold)
        logger.info('set constant question-question similarity threshold=%s',
                    q_q_threshold)
        method.set_q_q_threshold(q_q_threshold)
    result = []
    for x in range(0, 11, 1):
        threshold = x / 10.0
        # q_a_threshold固定
        if scheme == 1:
            logger.info('set question-question similarity thresholds=%s',
                        threshold)
            method.set_q_q_threshold(threshold)
            file_name = 'data/q-q-{}-q-a-{}.txt'.format(threshold, q_a_threshold)
        # q_q_threshold固定
        elif scheme == 2:
            logger.info('set question-answer similarity thresholds=%s',
                        threshold)
            method.set_q_a_threshold(threshold)
            file_name = 'data/q-q-{}-q-a-{}.txt'.format(q_q_threshold,
                                                        threshold)
        else:
            logger.info('set all similarity thresholds=%s',
                        threshold)
            method.set_q_a_threshold(threshold)
            method.set_q_q_threshold(threshold)
            file_name = 'data/q-q-{0}-q-a-{0}.txt'.format(threshold)
        if os.path.isfile(file_name):
            logger.info('%s exists', file_name)
        else:
            test(method, file_name=file_name)
        evaluation_result = evaluation(file_name=file_name)
        result.append({'threshold': threshold, 'result': evaluation_result})
    with codecs.open(output_name, encoding='utf-8', mode='wb') as f:
        f.write(json.dumps(result))


class DataSetGenerator():
    def __init__(self, data_set_file_name='data/test-set.txt',
                 result_file_name='data/actual-result.txt'):
        self.result_pattern = u'{}{}:{}\n'
        self.queue = Queue()
        self.question_num = 1
        self.answer_num = 1
        self.data_set_file_name = data_set_file_name
        self.result_file_name = result_file_name

    def generate_paragraph(self, paragraph):
        paragraph_lines = [self.result_pattern.format('Q', self.question_num, paragraph.question.title)]
        result_lines = [self.result_pattern.format('Q', self.question_num, 'N')]
        self.question_num += 1
        for reply in paragraph.replies:
            if reply.is_deleted == 1:
                continue
            if reply.is_question():
                test_line = self.result_pattern.format('Q',  self.question_num, reply.content)
                result_line = self.result_pattern.format('Q', self.question_num, 'F')
                result_lines.append(result_line)
                self.question_num += 1
            else:
                test_line = self.result_pattern.format('A', self.answer_num, reply.content)
                self.answer_num += 1
            paragraph_lines.append(test_line)
        return paragraph_lines, result_lines

    def generate(self):
        previous_category_id = None
        with codecs.open(self.data_set_file_name, encoding='utf-8', mode='wb') as data_set:
            with codecs.open(self.result_file_name, encoding='utf-8', mode='wb') as result:
                logger.info('start to generate data set')
                for paragraph in Session.query(Paragraph).filter(Paragraph.paragraph_id <= 500).filter(Paragraph.is_deleted == 0).all():
                    # 当前分类与上一话段分类一样，先不写，入队列
                    if paragraph.question.category_id == previous_category_id:
                        logger.info('same category id, put %s into queue',
                                    paragraph.paragraph_id)
                        self.queue.put(paragraph)
                        continue
                    paragraph_lines, result_lines = self.generate_paragraph(paragraph)
                    paragraph_lines.append('\n')
                    data_set.writelines(paragraph_lines)
                    result.writelines(result_lines)
                    previous_category_id = paragraph.question.category_id
                    # 看队列头是否能写
                    if not self.queue.empty():
                        paragraph_in_queue = self.queue.get()
                        if paragraph_in_queue.question.category_id == previous_category_id:
                            logger.info('same category id again, put %s into queue', paragraph_in_queue.paragraph_id)
                            self.queue.put(paragraph_in_queue)
                        else:
                            logger.info('dequeue')
                            paragraph_lines, result_lines = self.generate_paragraph(paragraph_in_queue)
                            paragraph_lines.append('\n')
                            data_set.writelines(paragraph_lines)
                            result.writelines(result_lines)
                            previous_category_id = paragraph_in_queue.question.category_id
                while not self.queue.empty():
                    paragraph_in_queue = self.queue.get()
                    paragraph_lines, result_lines = self.generate_paragraph(paragraph_in_queue)
                    paragraph_lines.append('\n')
                    data_set.writelines(paragraph_lines)
                    result.writelines(result_lines)
                logger.info('finished generating data set')


def show_usage():
    pass


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'htdea', ['help', 'test-set', 'de-boni', 'evaluation', 'adjust-threshold'])
    except getopt.GetoptError:
        show_usage()
        return
    logging.config.dictConfig(LOGGING)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            show_usage()
            return
        elif opt in ('-t', '--test-set'):
            data_set_generator = DataSetGenerator()
            data_set_generator.generate()
        elif opt in ('-d', '--de-boni'):
            build_word_embedding_vectors()
            test()
        elif opt in ('-e', '--evaluation'):
            evaluation()
        elif opt in ('-a', '--adjust-threshold'):
            essentials = build_essentials('data')
            word_embedding_vectors = build_word_embedding_vectors()
            method_config = {
                'essentials': {
                    'third_person_pronoun_dict': essentials['third_person_pronoun_dict'],
                    'demonstrative_pronoun_dict': essentials['demonstrative_pronoun_dict'],
                    'cue_word_dict': essentials['cue_word_dict'],
                    'stop_word_dict': essentials['stop_word_dict'],
                },
                'word_similarity_calculators': {
                    'how_net': {
                        'class': 'HowNetCalculator',
                        'word_similarity_table': None
                    },
                    'word_embedding': {
                        'class': 'WordEmbeddingCalculator',
                        'word_embedding_vectors': word_embedding_vectors
                    }
                },
                'method': {
                    'de_boni': {
                        'class': 'DeBoni',
                        'sentence_similarity_calculator': {
                            'word_similarity_calculator': 'how_net'
                        }
                    }
                }
            }
            paragraph_recognition.configure(method_config)
            # for x in range(0, 11, 1):
            #     threshold = x / 10.0
            #     adjust_threshold(q_a_threshold=threshold)
            adjust_threshold()


if __name__ == '__main__':
    main(sys.argv[1:])
