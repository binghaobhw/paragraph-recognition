#!/usr/bin/env python
# coding: utf-8

import codecs
from glob import glob
import os
import unittest
from mock import Mock
import experiment
import method


class TestDatasetGenerator(unittest.TestCase):
    def test_generate(self):
        dataset_filename = 'data/test-set.test'
        label_filename = 'data/label.test'
        dataset_generator = experiment.DatasetGenerator(dataset_filename, label_filename)
        dataset_generator.generate(2)
        self.assertTrue(os.path.isfile(dataset_filename))
        self.assertTrue(os.path.isfile(label_filename))
        with codecs.open(label_filename, encoding='utf-8') as f:
            a = 0
            for line in f:
                a += 1
            self.assertEqual(a, 4)
        os.remove(dataset_filename)
        os.remove(label_filename)


class TestTrainData(unittest.TestCase):
    def test_train_data(self):
        mock_method = Mock()
        mock_method.feature_names = [u'pronoun', u'proper_noun', u'noun',
                                     u'verb', u'max_sentence_similarity']
        mock_method.features.return_value = [True, True, False, False, 0.5555]
        dataset_filename = 'data/test-set.test'
        label_filename = 'data/label.test'
        train_set_filename = 'data/train-set.test'
        with codecs.open(dataset_filename, encoding='utf-8', mode='wb') as d:
            d.write(u"Q1:乳房的触诊顺序是什么\n"
                    u"A1:你男女？\n"
                    u"Q2:这和问题有关系吗\n"
                    u"A2:看乳房是啥病\n"
                    u"\n"
                    u"Q3:有没有人知道网店管家这款软件，怎么样？\n"
                    u"A3:可以\n"
                    u"Q4:你知道\n"
                    u"\n"
                    u"Q5:买了一双詹姆斯11涂鸦 但是不知道是不是xdr底的 没有吊牌 现在很担心 只要照片可以看出来是不\n"
                    u"A4:盒子还在吗\n"
                    u"Q6:在的话应该怎么看\n"
                    u"\n")
        with codecs.open(label_filename, encoding='utf-8', mode='wb') as l:
            l.write(u'Q1:0\n'
                    u'Q2:1\n'
                    u'Q3:0\n'
                    u'Q4:1\n'
                    u'Q5:0\n'
                    u'Q6:1\n')
        experiment.generate_train_data(mock_method, dataset_filename, label_filename,
                   train_set_filename)
        self.assertTrue(os.path.isfile(dataset_filename))
        self.assertTrue(os.path.isfile(label_filename))
        self.assertTrue(os.path.isfile(train_set_filename))
        with codecs.open(train_set_filename, encoding='utf-8') as f:
            a = 0
            for line in f:
                a += 1
            self.assertEqual(a, 7)
        os.remove(dataset_filename)
        os.remove(label_filename)
        os.remove(train_set_filename)


class TestKFoldCross(unittest.TestCase):
    def test_k_fold_cross(self):
        method.configure(experiment.method_config)
        result = experiment.k_fold_cross(2, 10, 'my_method', 'my-method')
        self.assertEqual(len(result), 3)
        file_pattern = 'data/2-fold-cross-*'
        for f in glob(file_pattern):
            os.remove(f)


class TestAnalyzeFeature(unittest.TestCase):
    def test_analyze_feature(self):
        method.configure(experiment.method_config)
        feature_names = method.ImprovedMethod.feature_names
        expectation = {'origin': 0}
        i = 0
        for feature_name in feature_names:
            i += 1
            expectation[feature_name] = i
        mock_k_fold_cross = Mock()
        mock_k_fold_cross.side_effect = range(0, 9)
        experiment.k_fold_cross = mock_k_fold_cross
        result = experiment.analyze_feature(2, 10, 'my_method')
        self.assertDictEqual(result, expectation)


class TestKFoldCrossDataset(unittest.TestCase):
    def test_k_fold_cross_dataset(self):
        method.configure(experiment.method_config)
        result = experiment.k_fold_cross_dataset(2, 10)
        self.assertEqual(len(result), 2)
        for i in result:
            for f in result[i].itervalues():
                os.remove(f)


class TestAnalyze(unittest.TestCase):
    def test_analyze(self):
        # result = experiment.analyze(u'谢谢你')
        result = experiment.analyze(u'多媒体的英文单词是Multimedia，它由media和multi两部分组成。一般理解为多种媒体的综合。多媒体技术不是各种信息媒体的简单复合，它是一种把文本(Text)、图形(Graphics)、图像(Images)、动画(Animation)和声音(Sound)等形式的信息结合在一起，并通过计算机进行综合处理和控制，能支持完成一系列交互式操作的信息技术。多媒体技术的发展改变了计算机的使用领域，使计算机由办公室、实验室中的专用品变成了信息社会的普通工具，广泛应用于工业生产管理、学校教育、公共信息咨询、商业广告、军事指挥与训练，甚至家庭生活与娱乐等领域。多媒体技术有以下几个主要特点：（1）集成性 能够对信息进行多通道统一获取、存储、组织与合成。（2）控制性 多媒体技术是以计算机为中心，综合处理和控制多媒体信息，并按人的要求以多种媒体形式表现出来，同时作用于人的多种感官。 （3）交互性 交互性是多媒体应用有别于传统信息交流媒体的主要特点之一。传统信息交流媒体只能单向地、被动地传播信息，而多媒体技术则可以实现人对信息的主动选择和控制。（4）非线性 多媒体技术的非线性特点将改变人们传统循序性的读写模式。以往人们读写方式大都采用章、节、页的框架，循序渐进地获取知识，而多媒体技术将借助超文本链接（Hyper Text Link）的方法，把内容以一种更灵活、更具变化的方式呈现给读者。（5）实时性 当用户给出操作命令时，相应的多媒体信息都能够得到实时控制。（6）信息使用的方便性 用户可以按照自己的需要、兴趣、任务要求、偏爱和认知特点来使用信息，任取图、文、声等信息表现形式。（7）信息结构的动态性 “多媒体是一部永远读不完的书”，用户可以按照自己的目的和认知特征重新组织信息，增加、删除或修改节点，重新建立链1．2 文件表示媒体的各种编码数据在计算机中都是以文件的形式存储的，是二进制数据的集合。文件的命名遵循特定的规则，一般由主名和扩展名两部分组成，主名与扩展名之间用"．"隔开，扩展名用于表示文件的格式类型。1．3 多媒体信息的类型及特点（1）文本 文本是以文字和各种专用符号表达的信息形式，它是现实生活中使用得最多的一种信息存储和传递方式。用文本表达信息给人充分的想象空间，它主要用于对知识的描述性表示，如阐述概念、定义、原理和问题以及显示标题、菜单等内容。（2）图像 图像是多媒体软件中')
        self.assertTrue(isinstance(result, list))


if __name__ == '__main__':
    unittest.main()
