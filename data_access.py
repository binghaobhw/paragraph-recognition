#!/usr/bin/env python
# coding: utf-8

__author__ = 'wilfredwang'
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import func
from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mysql import (BIGINT,
                                       CHAR,
                                       DATETIME,
                                       TINYINT,
                                       INTEGER,
                                       VARCHAR)

engine = create_engine(
    'mysql://root:wangbinghao@127.0.0.1:3306/test?charset=utf8',
    encoding='utf-8', pool_size=10)

session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

Base = declarative_base()


class Question(Base):
    __tablename__ = 'zhidao_question'

    question_id = Column(BIGINT(unsigned=True), primary_key=True)
    category_id = Column(INTEGER(unsigned=True))
    title = Column(VARCHAR(1000))
    created_time = Column(DATETIME, default=func.now())
    modified_time = Column(DATETIME, default=func.now())
    is_deleted = Column(TINYINT, default=0)

    def __init__(self, question_id, category_id=None, title=None):
        self.question_id = question_id
        self.category_id = category_id
        self.title = title

    def __repr__(self):
        return "<Question(question_id='%s', category_id='%s', title='%s')>" \
               % (self.question_id, self.category_id, self.title)

    def __str__(self):
        return self.__repr__()


class Paragraph(Base):
    __tablename__ = 'zhidao_paragraph'

    paragraph_id = Column(BIGINT(unsigned=True), primary_key=True)
    question_id = Column(BIGINT(unsigned=True),
                         ForeignKey('zhidao_question.question_id'))
    created_time = Column(DATETIME, default=func.now())
    modified_time = Column(DATETIME, default=func.now())
    is_deleted = Column(TINYINT, default=0)

    replies = relationship('Reply', order_by='Reply.reply_id')
    question = relationship('Question')

    def __init__(self, question_id):
        self.question_id = question_id

    def __repr__(self):
        return "<Paragraph(paragraph_id='%s', question_id='%s', reply='%s')>" \
               % (self.paragraph_id, self.question_id, self.replies)

    def __str__(self):
        return self.__repr__()


class Reply(Base):
    __tablename__ = 'zhidao_reply'

    reply_id = Column(BIGINT(unsigned=True), primary_key=True)
    paragraph_id = Column(BIGINT(unsigned=True),
                          ForeignKey('zhidao_paragraph.paragraph_id'))
    type = Column(INTEGER)
    content = Column(VARCHAR(10000))
    created_time = Column(DATETIME, default=func.now())
    modified_time = Column(DATETIME, default=func.now())
    is_deleted = Column(TINYINT, default=0)

    def __init__(self, type, content):
        self.type = type
        self.content = content

    def __repr__(self):
        return "<Reply(reply_id='%s', paragraph_id='%s', type='%s', " \
               "content='%s')>" \
               % (self.reply_id, self.paragraph_id, self.type, self.content)

    def __str__(self):
        return self.__repr__()


class LtpResult(Base):
    __tablename__ = 'ltp_result'

    md5 = Column(CHAR(32), primary_key=True)
    json_text = Column(VARCHAR(10000))
    created_time = Column(DATETIME, default=func.now())
    modified_time = Column(DATETIME, default=func.now())
    is_deleted = Column(TINYINT, default=0)

    def __init__(self, md5, json_text):
        self.md5 = md5
        self.json_text = json_text

    def __repr__(self):
        return "<LtpResult(md5='%s', json_text='%s')>" \
               % (self.md5, self.json_text)

    def __str__(self):
        return self.__repr__()
