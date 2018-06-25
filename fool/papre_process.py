#/usr/bin/env python
#-*-coding:utf-8-*-


"""
完成分词，词性标注，实体替换，去掉停用词等
"""

import jieba
from jieba import posseg

import re
import codecs

PATTERN_MAPPRING={
    u"\*":"",
    u"蚂蚁花呗":u"花呗",
    u"蚂蚁借呗":u"借呗",
}

def load_stop_words(file_path):
    stop_words = set()
    with codecs.open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip("\n").strip()
            if not line:
                continue
            stop_words.add(line)
    return list(stop_words)

class PreResult(object):
    def __init__(self,raw_text, raw_words, clean_text, clean_words, tag_words):
        self.raw_words = raw_words
        self.raw_text = raw_text
        self.clean_text = clean_text
        self.clean_words = clean_words
        self.tag_words = tag_words


class PapreProcessor(object):
    def __init__(self, user_dict, stop_words_dict=None):
        self.user_dict = user_dict
        self.stop_words = []
        self.tag_list = ["n", "v", 'a']
        jieba.load_userdict(user_dict)
        if stop_words_dict:
            self.stop_words = load_stop_words(stop_words_dict)

    def cut_text(self,text):
        words = list(jieba.cut(text))
        return words

    def _delete_stopwords(self, words):
        words = [w for w in words if w not in self.stop_words]
        return words

    def _replace_pattern(self, text):
        for k, v in PATTERN_MAPPRING.items():
            text = re.sub(re.compile(k), v, text)
        return text

    def _pos_tag(self, text):
        tag_res = list(posseg.cut(text))
        return tag_res

    def filter_by_tag(self, text):
        tag_result = self._pos_tag(text)
        words = [item.word for item in tag_result if item.flag in self.tag_list]
        return words

    def clean_text(self, text):
        text = self._replace_pattern(text)
        return text

    def process(self, text):
        clean_text = self.clean_text(text)
        words = self.cut_text(text)
        clean_words = self.cut_text(clean_text)
        clean_words = self._delete_stopwords(clean_words)
        tag_words = self.filter_by_tag(clean_text)
        result = PreResult(text, words, clean_text, clean_words, tag_words)
        return result



if __name__ == '__main__':
    processor =PapreProcessor("../data/user_dict", "../data/stopwords.txt")
    source_file = "../data/all_sent.txt"
    seg_outf = open("../data/seg_result.txt", 'w')
    with open(source_file) as f:
        for _, line in enumerate(f):
            line = line.strip("\n").strip()
            result = processor.process(line)
            print(result.clean_text)
            print(result.raw_words)
            print(result.clean_words)
            print(result.tag_words)
            input()
