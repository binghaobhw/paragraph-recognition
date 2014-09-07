#!/usr/bin/env python
# coding: utf-8

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
                                       VARCHAR,
                                       TEXT)

engine = create_engine(
    'mysql://root:wangbinghao@127.0.0.1:3306/test?charset=utf8',
    encoding='utf-8', pool_size=10)

session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

Base = declarative_base()


class Question(Base):
    __tablename__ = 'question'

    id = Column(BIGINT(unsigned=True), primary_key=True)
    category_id = Column(INTEGER(unsigned=True))
    content = Column(VARCHAR(1000))
    created_time = Column(DATETIME, default=func.now())
    modified_time = Column(DATETIME, default=func.now())
    is_deleted = Column(TINYINT, default=0)

    def __init__(self, id_, category_id=None, content=None):
        self.id = id_
        self.category_id = category_id
        self.content = content

    def __repr__(self):
        return "<Question(id='%s', category_id='%s', content='%s')>" \
               % (self.id, self.category_id, self.content)

    def __str__(self):
        return self.__repr__()


class Paragraph(Base):
    __tablename__ = 'paragraph'

    id = Column(BIGINT(unsigned=True), primary_key=True)
    question_id = Column(BIGINT(unsigned=True),
                         ForeignKey('question.id'))
    created_time = Column(DATETIME, default=func.now())
    modified_time = Column(DATETIME, default=func.now())
    is_deleted = Column(TINYINT, default=0)

    replies = relationship('Reply', order_by='Reply.id')
    question = relationship('Question')

    def __init__(self, question_id):
        self.question_id = question_id

    def __repr__(self):
        return "<Paragraph(id='%s', question_id='%s', reply='%s')>" \
               % (self.id, self.question_id, self.replies)

    def __str__(self):
        return self.__repr__()


class Reply(Base):
    __tablename__ = 'reply'

    id = Column(BIGINT(unsigned=True), primary_key=True)
    paragraph_id = Column(BIGINT(unsigned=True),
                          ForeignKey('paragraph.id'))
    type = Column(INTEGER)
    content = Column(VARCHAR(10000))
    created_time = Column(DATETIME, default=func.now())
    modified_time = Column(DATETIME, default=func.now())
    is_deleted = Column(TINYINT, default=0)

    def __init__(self, type_, content):
        self.type = type_
        self.content = content

    def __repr__(self):
        return "<Reply(id='%s', paragraph_id='%s', type='%s', " \
               "content='%s')>" \
               % (self.id, self.paragraph_id, self.type, self.content)

    def __str__(self):
        return self.__repr__()

    def is_question(self):
        return self.type == 0


class LtpResult(Base):
    __tablename__ = 'ltp_result'

    md5 = Column(CHAR(32), primary_key=True)
    json_text = Column(TEXT)
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


class FilteredParagraph(Base):
    __tablename__ = 'filtered_paragraph'

    id = Column(BIGINT(unsigned=True), primary_key=True)
    paragraph_id = Column(BIGINT(unsigned=True),
                          ForeignKey('zhidao_paragraph.id'))
    title = Column(VARCHAR(1000))

    paragraph = relationship('Paragraph')

    def __repr__(self):
        return "<FilteredParagraph(id='%s', id='%s', title='%s')>" \
               % (self.id, self.paragraph_id, self.title)

    def __str__(self):
        return self.__repr__()