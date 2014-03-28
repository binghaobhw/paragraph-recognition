#!/usr/bin/env python
# coding: utf-8
__author__ = 'wilfredwang'
from data_access import Session
from data_access import Paragraph
import traceback


with open('baidu-zhidao-paragraph.txt', 'wb') as f:
    count = 0
    try:
        for paragraph in Session.query(Paragraph):
            lines = [paragraph.question.title, '\n']
            for reply in paragraph.reply:
                lines.append(reply.content)
                lines.append('\n')
            lines.append('\n')
            f.writelines([s.encode('utf-8') for s in lines])
            count += 1
        print count
    except:
        print 'error, count %d' % count
        traceback.print_exc()
