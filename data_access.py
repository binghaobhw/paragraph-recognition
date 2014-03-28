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
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import VARCHAR

engine = create_engine(
    'mysql://root:wangbinghao@127.0.0.1:3306/test?charset=utf8',
    encoding='utf-8')

session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

Base = declarative_base()


class Question(Base):
    __tablename__ = 'zhidao_question'

    question_id = Column(BIGINT(unsigned=True), primary_key=True)
    category_id = Column(INTEGER(unsigned=True))
    title = Column(VARCHAR(200))
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

    reply = relationship('Reply', order_by='Reply.reply_id')
    question = relationship('Question')

    def __init__(self, question_id):
        self.question_id = question_id

    def __repr__(self):
        return "<Paragraph(paragraph_id='%s', question_id='%s', reply='%s')>" \
               % (self.paragraph_id, self.question_id, self.reply)

    def __str__(self):
        return self.__repr__()


class Reply(Base):
    __tablename__ = 'zhidao_reply'

    reply_id = Column(BIGINT(unsigned=True), primary_key=True)
    paragraph_id = Column(BIGINT(unsigned=True),
                          ForeignKey('zhidao_paragraph.paragraph_id'))
    type = Column(INTEGER)
    content = Column(VARCHAR(200))
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
