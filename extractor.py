# coding:utf-8
import time
from Queue import Queue
import requests
import re
from threading import (Thread,
                       Event)
import logging
import logging.config
from urllib import quote
from os.path import isfile
from bs4 import BeautifulSoup
import random

from data_access import (Session,
                         Question,
                         Paragraph,
                         Reply)
from unicode_csv import (to_unicode,
                         read_csv,
                         write_csv)
from log_config import LOGGING, LOG_PROJECT_NAME

logger = logging.getLogger(LOG_PROJECT_NAME + '.extractor')

TOP_CATEGORY_CSV = 'top-category.csv'
SUB_CATEGORY_CSV = 'sub-category.csv'
NUM_RE = r'\d+'
HEADERS = {
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
'Accept-Encoding': 'gzip,deflate,sdch',
'Accept-Language': 'en,zh-CN;q=0.8,zh;q=0.6,zh-TW;q=0.4',
'Cache-Control': 'max-age=0',
'Connection': 'keep-alive',
'Referer': 'https://www.google.com.hk/',
'User-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.146 Safari/537.36'}
TIMEOUT = 5
browser = requests.session()


class Extractor(Thread):
    domain = 'http://zhidao.baidu.com'

    def __init__(self, name, queue, exit_signal):
        super(Extractor, self).__init__(name=name)
        self.queue = queue
        self.exit_signal = exit_signal

    def get_page(self, target, delay=False):
        if target is None:
            return None
        target_url = self.domain + target if target.startswith('/') else target
        if delay:
            sleep()
        logger.info('start to get %s', target)
        try:
            response = browser.get(target_url, timeout=TIMEOUT, headers=HEADERS)
        except:
            logger.error('fail to get %s', target_url, exc_info=True)
            return None
        logger.info('finished getting page')
        if response.status_code == requests.codes.ok:
            return BeautifulSoup(response.content)
        else:
            logger.info('bad response status, %s', response.status_code)
            return None

    def run(self):
        self.do_run()
        self.cleanup()
        logger.info('exit')

    def do_run(self):
        target_list = self.get_target_list()
        if target_list is None:
            return
        logger.info('start to traverse target list')
        for target in target_list:
            if self.exit_signal.isSet():
                logger.info('receive exit signal')
                return
            self.extract(target)
        logger.info('finished traversing target list')

    def get_target_list(self):
        pass

    def extract(self, target):
        pass

    def cleanup(self):
        pass


def get_next_page_link(page):
    if page is not None:
        next_page_anchor = page.find('a', 'pager-next')
        if next_page_anchor:
            return to_unicode(next_page_anchor['href'])
    return None


class CategoryExtractor(Extractor):
    def do_run(self):
        if isfile(TOP_CATEGORY_CSV):
            logger.info("'%s' exists, skip top category extraction",
                        TOP_CATEGORY_CSV)
        else:
            self.extract_top_category()
        if isfile(SUB_CATEGORY_CSV):
            logger.info("'%s' exists, skip sub category extraction",
                        SUB_CATEGORY_CSV)
        else:
            super(CategoryExtractor, self).do_run()

    def get_target_list(self):
        target_list = []
        try:
            csv = read_csv(TOP_CATEGORY_CSV)
        except:
            logger.error("fail to read '%s'", TOP_CATEGORY_CSV, exc_info=True)
        else:
            for name, url in csv:
                target_list.append(url)
        return target_list

    def extract_top_category(self):
        target = '/browse'
        page = self.get_page(target)
        if page is None:
            return
        logger.info('start to extract top category in %s', target)
        category_url_list = []
        for category_item in page.find_all('li', 'category-item'):
            anchor = category_item.a
            if anchor is None:
                continue
            category_name = to_unicode(anchor.string)
            category_url = to_unicode(anchor['href'])
            if category_name and category_url:
                category_url_list.append((category_name, category_url))
        if len(category_url_list) == 0:
            logger.error('no category found')
            return
        write_csv(TOP_CATEGORY_CSV, 'wb', category_url_list)
        logger.info("finished extracting top category into '%s'", TOP_CATEGORY_CSV)

    def extract(self, target):
        page = self.get_page(target)
        if page is None:
            return
        logger.info('start to extract sub category in %s', target)
        category_list = []
        for category_item in page.find_all('li', 'category-item'):
            for anchor in category_item.find_all('a'):
                name = to_unicode(anchor.string)
                if name is None:
                    name = to_unicode(name.strings)
                name = re.sub(r'\xa0.+', '', name)
                url = to_unicode(anchor['href'])
                category_list.append((name, url))
        write_csv(SUB_CATEGORY_CSV, 'ab', category_list)
        logger.info('finished extracting sub category in %s', target)


class CategoryPageExtractor(Extractor):

    def get_target_list(self):
        target_list = []
        try:
            csv = read_csv(TOP_CATEGORY_CSV)
        except:
            logger.error("fail to read '%s', set exit signal",
                         TOP_CATEGORY_CSV,  exc_info=True)
            self.exit_signal.set()
        else:
            for name, url in csv:
                target_list.append(url + '?lm=4')
        return target_list

    def extract(self, target):
        """Get unsolved question url whose answer num > 0
        :param target:
        """
        page = self.get_page(target)
        while not self.exit_signal.isSet() and page is not None:
            for answer_num_div in page.find_all(
                    'div', 'f-12 f-light question-answer-num'):
                matched_result = re.findall(NUM_RE, answer_num_div.string)
                answer_num = int(matched_result[0]) if len(matched_result) > 0 else 0
                if answer_num == 0:
                    continue
                title_container_div = answer_num_div.find_previous_sibling(
                    'div', {'class': 'title-container'})
                question_anchor = title_container_div.find(
                    'a', {'class': 'question-title'})
                question_url = to_unicode(question_anchor['href'])
                logger.info("start to put '%s' into queue(%d), "
                            "%d answer", question_url, self.queue
                            .qsize(), answer_num)
                self.queue.put(question_url)
                logger.info("finished putting '%s' into queue(%d)",
                            question_url, self.queue.qsize())
            next_page_link = get_next_page_link(page)
            if next_page_link is not None:
                page = self.get_page(next_page_link)
            else:
                return


class SearchPageExtractor(Extractor):
    URL_PREFIX = 'http://zhidao.baidu.com/search?lm=0&site=0&sites=0_3_2_4&date=0&ie=gbk&word='

    def get_target_list(self):
        target_list = []
        try:
            csv = read_csv(SUB_CATEGORY_CSV)
        except:
            logger.error("fail to read '%s', set exit signal",
                         SUB_CATEGORY_CSV,  exc_info=True)
            self.exit_signal.set()
        else:
            for name, url in csv:
                target_list.append(self.URL_PREFIX + quote(name.encode(
                    'utf8'), safe=''))
        return target_list

    def extract(self, target):
        page = self.get_page(target)
        while not self.exit_signal.isSet() and page:
            for question_anchor in page.find_all('a', 'ti t-ie6'):
                question_url = to_unicode(question_anchor['href'])
                logger.info("start to put '%s' into queue(%d)",
                            question_url, self.queue.qsize())
                self.queue.put(question_url)
                logger.info("finished putting '%s' into queue(%d)",
                            question_url, self.queue.qsize())
            next_page_link = get_next_page_link(page)
            if next_page_link is not None:
                page = self.get_page(next_page_link)
            else:
                return


def get_title(page):
    title = None
    title_span = page.find('span', 'ask-title')

    if title_span is not None:
        title = title_span.string
        if title is None:
            anchor = title_span.find('a', 'g-zhima-tag')
            if anchor is not None:
                title = anchor.next_sibling
    return to_unicode(title)


def is_visited(question_id):
    visited = True
    if question_id is not None:
        try:
            visited = (Session.query(Question).filter_by(
                question_id=question_id).count() != 0)
        except:
            logger.error('fail to query question_id %s', question_id,
                         exc_info=True)
    return visited


def sleep():
    second = random.randint(1, 600)
    logger.info('start to sleep %d', second)
    time.sleep(second)
    logger.info('wake up')


class ParagraphExtractor(Extractor):
    def do_run(self):
        while not self.exit_signal.isSet():
            logger.info('start to get question from queue(%d)',
                        self.queue.qsize())
            target = self.queue.get()
            logger.info("finished getting '%s' from queue(%d)",
                        target, self.queue.qsize())
            self.extract(target)

    def cleanup(self):
        Session.remove()

    def extract(self, target):
        logger.info('check whether visited')
        matched_result = re.findall(r'/(\d+).html', target)
        if len(matched_result) == 0:
            logger.error('invalid question page url %s', target)
            return
        question_id = matched_result[0]
        if is_visited(question_id):
            logger.info('%s is visited, skip', question_id)
            return
        page = self.get_page(target, delay=True)
        if page is None:
            logger.info('page is none, skip')
            return
        # save question
        anchor = page.find('a', {'alog-alias': 'qb-class-info'})
        if anchor is None:
            if page.find('title', text=u'百度--您的访问出错了') is None:
                logger.error('invalid question page %s', target)
            else:
                logger.error('auth page, set exit signal')
                self.exit_signal.set()
            return
        category_url = to_unicode(anchor['href'])
        category_id = re.findall(r'/(\d+)', category_url)[0]
        title = get_title(page)
        if title is None:
            logger.error('fail to get title in %s', target)
            return
        question = Question(question_id, category_id, title)
        Session.add(question)
        logger.info('start to insert %s', question)
        try:
            Session.commit()
        except:
            logger.error('fail to insert %s, rollback', question, exc_info=True)
            Session.rollback()
            return
        logger.info('finished inserting question')
        while not self.exit_signal.isSet() and page:
            for line_content_div in page.find_all('div', 'line content'):
                # answer only, skip
                if line_content_div.find('dt', 'ask f-12 grid') is None:
                    continue
                # generate paragraph
                paragraph = Paragraph(question_id)
                # generate reply
                a_content = line_content_div.find('pre', {'accuse': 'aContent'})
                if a_content is None:
                    logger.error('can not find aContent, structure changed')
                    break
                reply = to_unicode(a_content.strings)
                paragraph.replies.append(Reply(1, reply))
                for pre in line_content_div.find_all('pre'):
                    pre_accuse = pre.get('accuse', 'no')
                    if pre_accuse == 'aRA':
                        reply = to_unicode(pre.strings)
                        paragraph.replies.append(Reply(1, reply))
                    elif pre_accuse == 'qRA':
                        reply = to_unicode(pre.strings)
                        paragraph.replies.append(Reply(0, reply))
                Session.add(paragraph)
                logger.info('start to insert paragraph(%d replies)',
                            len(paragraph.replies))
                try:
                    Session.commit()
                except:
                    logger.error('fail to insert %s, rollback', paragraph,
                                 exc_info=True)
                    Session.rollback()
                logger.info('finished inserting paragraph')

            next_page_link = get_next_page_link(page)
            page = self.get_page(next_page_link, delay=True)
        logger.info('finished extracting paragraph in %s', target)

THREAD_NUM = 7
QUEUE_SIZE = 3


def main():
    logging.config.dictConfig(LOGGING)

    exit_signal = Event()
    category_extractor = CategoryExtractor('category_extractor',
                                           None, exit_signal)

    url_queue = Queue(QUEUE_SIZE)
    category_page_extractor = CategoryPageExtractor('category_page_extractor',
                                                    url_queue, exit_signal)
    # extractors = [category_page_extractor]
    search_page_extractor = SearchPageExtractor('search_page_extractor',
                                                url_queue, exit_signal)
    extractors = [category_page_extractor, search_page_extractor]
    for i in range(0, THREAD_NUM):
        extractors.append(ParagraphExtractor('paragraph_extractor_%d' % i,
                                             url_queue, exit_signal))
    logger.info('start all')
    try:
        category_extractor.start()
        category_extractor.join()
        for extractor in extractors:
            extractor.start()
        while not exit_signal.isSet():
            sleep()
    except KeyboardInterrupt:
        logger.info('receive ctrl-c ')
    except:
        logger.error('error occurred, exit', exc_info=True)
    finally:
        exit_signal.set()
        for extractor in extractors:
            extractor.join()
        logger.info('exit')


if __name__ == '__main__':
    main()
