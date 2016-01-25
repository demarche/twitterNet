# coding: utf-8

from gensim import corpora, models
import numpy as np
from numpy import random
random.seed(555)

import MeCab
import sys
import codecs
import os
from progressbar import ProgressBar

class MyTexts:
    def __init__(self, text_list):
        self.text_list = text_list

    def __iter__(self):
        #size = float()
        p = ProgressBar(max_value = len(self.text_list), min_value = 1)
        for i, line in enumerate(self.text_list):
            #if i%100 == 0:
            p.update((i+1))
            sentence = line.rstrip()

            tagger = MeCab.Tagger('')  # 別のTaggerを使ってもいい
            #print sentence
            node = tagger.parseToNode(sentence)
            words = []
            while node:
                # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
                info = node.feature.split(",")

                try:
                    word = node.surface
                except:
                    pass
                if word != 'EOS' and word != '' and info[0] != "記号":
                    if info[6] != '*':
                        words.append(info[6])
                    else:
                        words.append(word)

                node = node.next
            #print(words)
            yield words
        p.finish()


class LabeledLineSentence(object):
    def __init__(self, texts_words):
        self.texts_words = texts_words
        
    def __iter__(self):
        for uid, line in enumerate(self.texts_words):
            #print line
            tmp_model = models.doc2vec.TaggedDocument(line, [uid])
            yield tmp_model


if __name__=='__main__':
    ifname = 'C:\\doc2vec_result\\CP.txt'

    model_basename = 'model'

    comment_data = []
    comment_data_apd = comment_data.append
    with open(ifname, 'r', encoding="utf-8", errors="ignore") as f:
        for line in f:
            comment = line.strip()
            comment_data_apd(comment.strip())

    print(str(len(comment_data))+" corpus loaded")
    texts = MyTexts(comment_data)
    print("labeled line sentence")

    sentences = LabeledLineSentence(texts)

    print('start training')
    '''
    model = models.Doc2Vec(size=1000, alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    print('build vocab')
    model.build_vocab(sentences)

    # store the model to mmap-able files
    model.save(model_basename+'.d2v')
    '''
    model = models.Doc2Vec.load(str(5)+"\\"+model_basename+'.d2v')
    epoch = 10
    for i in range(epoch):
        print('iteration:' + str(i))
        try:
            os.mkdir(str(i))
        except:
            continue
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(str(i)+"\\"+model_basename+'.d2v')
    print('done training')

    model.save(model_basename+'.d2v')

