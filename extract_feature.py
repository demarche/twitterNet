import sys, os, os.path
import numpy as np
import glob
import MeCab
import pickle
from progressbar import ProgressBar
from gensim import models

from PIL import Image

class saver:
    def __init__(self, ROOT_PATH, in_size):
        model_basename = 'Data/model'
        print('loading d2v model..')
        self.model = models.Doc2Vec.load(model_basename+'.d2v')
        print('loaded')

        self.ROOT = ROOT_PATH
        self.lines = open(self.ROOT+'/user_info.txt', 'r').readlines()

        self.in_size = in_size

    def mymecab(self, line):
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
        return words

    def preprocess(self, pil_image):
        pil_image = pil_image.resize((self.in_size, self.in_size), Image.BILINEAR)
        pil_image = pil_image.convert('RGB')
        in_ = np.asarray(pil_image, dtype='f')
        in_ = in_.transpose(2, 0, 1)
        in_ = in_[::-1]
        return in_

    def __iter__(self):
        mean_image = pickle.load(open("image_mean.pkl", "rb"))

        for line in self.lines:
            # get user info
            info = line.split(",")
            user_id = info[0].replace("\ufeff", "")

            # read corpus
            corpus_path = self.ROOT+'/corpus/'+user_id+'.txt'
            cp_lines = open(corpus_path, 'r', encoding="utf-8", errors="ignore").readlines()

            for cp in cp_lines:
                cp_elem = cp.split(",")
                # get answer
                RT = float(cp_elem[2])

                image_folder = self.ROOT+'/images/'+user_id+'/'+cp_elem[1]+'/*'

                for image_path in glob.glob(image_folder):
                    # save image
                    try:
                        img = self.preprocess(Image.open(image_path))
                        img -= mean_image
                    except:
                        continue

                    # extract corpus feature
                    words = self.mymecab(cp_elem[0])
                    cp_feat = self.model.infer_vector(words)

                    yield img, cp_feat, RT

    def save(self):
        # makedir
        saving_folder_name = "images"
        try:
            os.makedirs(saving_folder_name)
        except:
            pass

        saving_image_name = 0
        corpus_features = []
        answers = []
        answers_RT = []
        mean_image = np.zeros((3, self.in_size, self.in_size))

        p = ProgressBar(max_value = len(self.lines), min_value = 0)
        for i, line in enumerate(self.lines):
            # update
            p.update(i+1)

            # get user info
            info = line.split(",")
            user_id = info[0].replace("\ufeff", "")
            RT_dev = float(info[9])

            # read corpus
            corpus_path = self.ROOT+'/corpus/'+user_id+'.txt'
            cp_lines = open(corpus_path, 'r', encoding="utf-8", errors="ignore").readlines()

            for cp in cp_lines:
                cp_elem = cp.split(",")

                # get answer
                RT = float(cp_elem[2])
                ans = "0"
                if RT != 0 and RT < RT_dev:
                    ans = "1"
                elif RT >= RT_dev and RT < RT_dev*2:
                    ans = "2"
                elif RT >= RT_dev*2:
                    ans = "3"

                image_folder = self.ROOT+'/images/'+user_id+'/'+cp_elem[1]+'/*'

                for image_path in glob.glob(image_folder):
                    # save image

                    try:
                        img = self.preprocess(Image.open(image_path))
                    except:
                        continue
                    pickle.dump(img, open(saving_folder_name+"/"+str(saving_image_name)+".pkl", "wb"))
                    saving_image_name += 1

                    # construct image mean
                    mean_image += img

                    # extract corpus feature

                    words = self.mymecab(cp_elem[0])
                    cp_feat = self.model.infer_vector(words)
                    corpus_features.append(cp_feat)

                    # save ans
                    answers.append(ans)
                    answers_RT.append(RT)
        mean_image /= len(answers)
        p.finish()

        print("save answrs")
        pickle.dump(answers, open("answers.pkl", "wb"))
        pickle.dump(answers_RT, open("answers_RT.pkl", "wb"))
        print("save image_mean")
        pickle.dump(mean_image, open("image_mean.pkl", "wb"))
        print("save corpus features")
        pickle.dump(corpus_features, open("corpus_features.pkl", "wb"))
