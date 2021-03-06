from chainer import Link, Chain, ChainList, computational_graph, cuda, FunctionSet, Variable
from GoogleNetBN import twitterNet_worker

import pickle
import six, sys
import numpy as np
import csv
import glob
from PIL import Image
from progressbar import ProgressBar
import pickle
import os.path
import argparse
from extract_feature import saver

in_size = 256

def get_images(perm, path):

    mean_image = pickle.load(open(os.path.join(path, "image_mean.pkl"), "rb"))
    images_path = os.path.join(path, "images")
    img_list=[]
    for img_name in perm:
        img = pickle.load(open(os.path.join(images_path, str(img_name)+".pkl"), "rb"))
        img -= mean_image
        img_list.append(img)
    return np.array(img_list, dtype=np.float32)

def build_imagesets(path):
    # path is ~moved/~
    my_saver = saver(path, in_size)
    my_saver.save()

def reduce_label(labels, k, split_perm):
    db_len = len(labels)
    lbmax = max(labels)+1

    isUse = [False]*db_len
    counter = [0]*lbmax
    print("reducing label...")
    p = ProgressBar(max_value=len(labels), min_value=1)
    for i, label in enumerate(labels):
        p.update(i+1)
        if abs(int(label)) > 600:# 絶対値600以上は消去
            continue
        if counter[int(label)] >= k:# k個以上のデータ数は省く
            continue
        isUse[i] = True
        counter[label] += 1
    split_perm = np.array([x for x in split_perm if isUse[x]])
    p.finish()
    return split_perm

def output_test(path, gpu_id, saved_path, regression, useImage, useDoc):
    gpu = gpu_id >= 0
    print("reg : ", regression)
    print("gpu : ", gpu_id)
    print("save path : ", saved_path)
    print("img : ", useImage)
    print("doc : ", useDoc)
    dir = os.path.dirname(saved_path)
    if gpu:
        cuda.check_cuda_available()

    print("loading labels..")
    if regression:
        labels = np.array(pickle.load(open(os.path.join(path, 'answers_RT2.pkl'), "rb")), dtype=np.float32)
    else:
        labels = np.array(pickle.load(open(os.path.join(path, "answers.pkl"), "rb")), dtype=np.int32)

    print("loading doc2vec model..")
    doc_vectors = np.array(pickle.load((open(os.path.join(path, "corpus_features.pkl"), "rb"))), dtype=np.float32)

    print("make NN model..")
    if regression:
        dim = 1
    else:
        dim = 4
    worker = twitterNet_worker(dim, [min(labels)])

    # for fname in glob.glob('/media/yamashita004/4dad8012-5855-4d11-8128-8fc5247ba677/NeuralNet/GoogleNetBN_REG/model/*.model' ):

    print("loading NN model..")
    worker.load(saved_path)
    #print(fname)
    #worker.load(fname)
    if gpu:
        cuda.get_device(gpu_id).use()
        worker.model.to_gpu()

    perm = pickle.load(open(os.path.join(dir, 'test_perm.pkl'), "rb"))
    batchsize = 30
    catans = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    coef=[]

    p = ProgressBar(max_value=len(perm), min_value=1)
    pred=[]
    ans=[]
    for i in six.moves.range(0, len(perm), batchsize):
        p.update(i+1)
        x_batch = get_images(perm[i:i + batchsize], path)
        x_batch_doc = doc_vectors[perm[i:i + batchsize]]
        y_batch = labels[perm[i:i + batchsize]]
        if regression:
            acoef,h,t=worker.test(x_batch, x_batch_doc, y_batch, regression, gpu=gpu)
            pred.extend(h)
            ans.extend(t)
            coef.append(acoef)
        else:
            for pred, ans in zip([h,t], y_batch):
                mymax = 0
                myid=0
                for i,x in enumerate(pred):
                    if mymax < x:
                        mymax = x
                        myid = i
                catans[ans][myid] += 1

    pickle.dump(catans, open(os.path.join(path, "catdic.pkl"), "wb"))
    p.finish()
    if regression:
        corr = np.corrcoef(pred, ans)[0, 1]
        print(corr)
        print(np.mean(coef))
        with open('some.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for h, t in zip(pred, ans):
                writer.writerow([h, t])

def create_split_perm(regression, labels, path):
    label_max = int(max(labels) + 1)
    db_len = len(labels)
    split_perm = np.random.permutation(db_len)
    if not regression:
        # 均衡化
        lbl_cnt = [labels.count(x) for x in range(label_max)]
        #open("before.csv", "w").write("\n".join(map(str, lbl_cnt)))
        print(lbl_cnt)
        k = min(lbl_cnt)
        print(k)
        print("reducing")
        split_perm = reduce_label(labels, k, split_perm)
        # check
        lbl_check=[]
        labels = np.array(labels)
        for i in range(label_max):
            lbl_check.append(list(labels[split_perm]).count(i))
        #open("after.csv", "w").write("\n".join(map(str, lbl_check)))
        print(lbl_check)
    else:
        if os.path.exists(os.path.join(path, "lbl_cnt.pkl")):
            print("load lbl_cnt")
            lbl_cnt = pickle.load(open(os.path.join(path, "lbl_cnt.pkl"), "rb"))
        else:
            print("create lbl_cnt")
            lbl_cnt = []
            p = ProgressBar(max_value=label_max, min_value=1)
            for i, x in enumerate(range(label_max)):
                p.update(i+1)
                lbl_cnt.append(labels.count(x))
            p.finish()
            pickle.dump(lbl_cnt, open(os.path.join(path, "lbl_cnt.pkl"), "wb"))

        print("create mean")
        #lbl_cnt_mean = np.mean(list(filter(lambda t:t>10, lbl_cnt)))
        lbl_cnt_mean=500
        print(lbl_cnt)
        print(lbl_cnt_mean)
        # reduce
        split_perm = reduce_label(labels, lbl_cnt_mean, split_perm)
        labels2 = [labels[x] for x in split_perm]
        lbl_cnt = [labels2.count(x) for x in range(label_max)]
        print(lbl_cnt)
        # remove sparse element
        """
        remove_list = []
        for i, x in list(enumerate(lbl_cnt))[::-1]:
            if x != 1 and x != 0:
                break
            elif i+1 == label_max or x == 1:
                remove_list.append(i)
            elif x == 1 and (lbl_cnt[i+1] != 0 or lbl_cnt[i-1] != 0):
                break
        split_perm = list(split_perm)
        for x in remove_list:
            index = labels.index(x)
            split_perm.remove(index)
        """
        split_perm = np.array(split_perm)

        labels2 = [labels[x] for x in split_perm]
        label_max = max(labels2) + 1
        lbl_cnt = [labels2.count(x) for x in range(label_max)]
        print(lbl_cnt)
        print(len(split_perm))
    return split_perm

def train_and_test(path, gpu_id, load_path, saved_path, regression, useImage, useDoc, iter):
    train_test_rate = 0.2
    batchsize = 25
    n_epoch = iter
    gpu = gpu_id >= 0
    print("reg : ", regression)
    print("gpu : ", gpu_id)
    print("path : ", path)
    print("load path : ", load_path)
    print("save path : ", saved_path)
    print("img : ", useImage)
    print("doc : ", useDoc)
    print("iter : ", iter)
    print("loading data")
    if regression:
        labels = pickle.load(open(os.path.join(path, 'answers_RT.pkl'), "rb"))
    else:
        labels = pickle.load(open(os.path.join(path, "answers.pkl"), "rb"))
    doc_vectors = pickle.load((open(os.path.join(path, "corpus_features.pkl"), "rb")))
    doc_vectors = np.array(doc_vectors, dtype=np.float32)

    labels = list(map(int, labels))
    label_max = max(labels) + 1

    # split train test data
    if saved_path == "":
        dir = ""
    else:
        dir = os.path.dirname(saved_path)
    print("dir is ", dir)
    if dir != "" and os.path.exists(os.path.join(dir, "train_perm.pkl")) and os.path.exists(os.path.join(dir, "test_perm.pkl")) and regression:
        train_perm = pickle.load(open(os.path.join(dir, "train_perm.pkl"), "rb"))
        test_perm = pickle.load(open(os.path.join(dir, "test_perm.pkl"), "rb"))
        print("loaded")
        N = len(train_perm)
        N_test = len(test_perm)
    else:
        print("normalarize")
        split_perm = create_split_perm(regression, labels, path)
        db_len=len(split_perm)
        N_test = int(train_test_rate*db_len)
        N = int(db_len-N_test)
        train_perm = split_perm[:N]
        test_perm = split_perm[N:]
        print("train:", len(train_perm))
        print("test:", len(test_perm))
        pickle.dump(train_perm, open(os.path.join(load_path, "train_perm.pkl"), "wb"))
        pickle.dump(test_perm, open(os.path.join(load_path, "test_perm.pkl"), "wb"))

    if regression:
        labels = np.array(labels, dtype=np.float32).reshape(len(labels), 1)
    else:
        labels = np.array(labels, dtype=np.int32)

    # make model
    print("make model")
    train_losses = []
    train_scores = []
    test_move = []
    if regression:
        worker = twitterNet_worker(1, min(labels))
    else:
        worker = twitterNet_worker(label_max, min(labels))
    if saved_path != "" and os.path.exists(saved_path):
        loaded_epoch = worker.load(saved_path)
        train_losses = pickle.load(open(os.path.join(load_path, "train_losses.pkl"), "rb"))[:loaded_epoch-1]
        train_scores = pickle.load(open(os.path.join(load_path, "train_scores.pkl"), "rb"))[:loaded_epoch-1]
        test_move = pickle.load(open(os.path.join(load_path, "test_score.pkl"), "rb"))[:int((loaded_epoch-1)/5)]
        print(len(train_losses))
        print(len(train_scores))
        print(len(test_move))
    else:
        loaded_epoch = 1
    if gpu:
        cuda.get_device(0).use()
        worker.model.to_gpu()

    for epoch in six.moves.range(loaded_epoch, n_epoch + 1):
        print('epoch', epoch)
        # training
        perm = np.random.permutation(N)
        sum_loss = 0
        p = ProgressBar(max_value=N, min_value=1)
        myh = [] #トレーニング用
        myt = []
        for i in six.moves.range(0, N, batchsize):
            p.update(i+1)
            x_batch = get_images(train_perm[perm[i:i + batchsize]], path)
            x_batch_doc = doc_vectors[train_perm[perm[i:i + batchsize]]]
            y_batch = labels[train_perm[perm[i:i + batchsize]]]
            # Pass the loss function (Classifier defines it) and its arguments
            loss, h, t = worker.train(x_batch, x_batch_doc, y_batch, regression=regression, gpu=gpu, useImage=useImage, useDoc=useDoc)
            sum_loss += float(loss) * len(y_batch)
            myh.extend(list(h))
            myt.extend(list(t))
        p.finish()
        if regression:
            pred = np.corrcoef(np.array(myh), np.array(myt))[0, 1]
        else:
            pred = np.mean(np.array(myh))
        print("train mean loss: %f" % (sum_loss / N))
        print("train mean corr: %f" % (pred))
        train_losses.append(sum_loss / N)
        train_scores.append(pred)
        pickle.dump(train_losses, open(os.path.join(load_path, "train_losses.pkl"), "wb"))
        pickle.dump(train_scores, open(os.path.join(load_path, "train_scores.pkl"), "wb"))
        # test
        if epoch % 5 == 0 and epoch != 0:
            sum_accuracy = [0.0] * label_max
            total_acc_elem = [0] * label_max
            sum_corr = 0
            myh = [] #テスト用
            myt = []
            p = ProgressBar(max_value=N_test, min_value=1)
            for i in range(0, N_test, batchsize):
                p.update(i+1)
                x_batch = get_images(test_perm[i:i + batchsize], path)
                x_batch_doc = doc_vectors[test_perm[i:i + batchsize]]
                y_batch = labels[test_perm[i:i + batchsize]]

                if regression:
                    h, t = worker.test(x_batch, x_batch_doc, y_batch, regression=regression, gpu=gpu, useImage=useImage, useDoc=useDoc)
                    myh.extend(list(h))
                    myt.extend(list(t))
                else:
                    # ラベルごとの精度を出す
                    acc = [0.0] * label_max
                    for label in range(label_max):
                        labeled_perm = [x for x in np.arange(len(y_batch)) if y_batch[x] == label]
                        if len(labeled_perm) != 0:
                            acc[label] = worker.test(x_batch[labeled_perm], x_batch_doc[labeled_perm], y_batch[labeled_perm],
                                             regression=regression, gpu=gpu, useImage=useImage, useDoc=useDoc)*len(labeled_perm)
                        total_acc_elem[label] += len(labeled_perm)
                    sum_accuracy = [s+float(t) for s, t in zip(sum_accuracy, acc)]
            p.finish()
            if regression:
                corr = np.corrcoef(np.array(myh), np.array(myt))[0, 1]
                print("test mean corr: %f" % (corr))
                test_move.append(corr)
                pickle.dump(test_move, open(os.path.join(load_path, "test_score.pkl"), "wb"))
            else:
                sum_accuracy = [t / float(u) for t, u in zip(sum_accuracy, total_acc_elem)]
                print("\n".join(map(str, sum_accuracy)))
                print("mean:", np.mean(sum_accuracy))
                test_move.append(np.mean(sum_accuracy))
                pickle.dump(test_move, open(os.path.join(load_path, "test_score.pkl"), "wb"))

        if epoch % 5 == 0 and epoch != 0:
            # Save the model and the optimizer
            print('save the model')
            worker.save(load_path, epoch)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process about tweetNet')
    parser.add_argument('-k', "--kind", dest='kind', default=0, type=int,
                        help='kind of process. 0: train and test. 1: test and output. 2: build image sets.  (default: 0)')
    parser.add_argument('-p', "--path", dest='path', default=os.path.abspath(os.path.dirname(__file__))+"\\data", type=str,
                        help='working path.  (default: my dir)')
    parser.add_argument('-s', "--saved_path", dest='saved_path', default="", type=str,
                        help='saved model path.  (default: empty)')
    parser.add_argument('-l', "--load_path", dest='load_path', default="", type=str,
                        help='load path.  (default: empty)')
    parser.add_argument('-g', "--gpu", dest='gpu_id', default=0, type=int,
                        help='using gpu id.  (default: 0)')
    parser.add_argument('-r', "--reg", dest='regression', type=int, default=0,
                        help='using regression predict.  (default: False)')
    parser.add_argument('-img', "--image", dest='useimage', type=int, default=1,
                        help='using image.  (default: True)')
    parser.add_argument('-doc', "--document", dest='usedoc', type=int, default=1,
                        help='using document.  (default: True)')
    parser.add_argument('-i', "--iter", dest='iter', type=int, default=1000,
                        help='iteration.  (default: 1000)')

    args = parser.parse_args()
    if args.kind == 0:
        print(args.path)
        print(args.saved_path)
        train_and_test(args.path, args.gpu_id, args.load_path, args.saved_path, args.regression == 1, args.useimage == 1, args.usedoc == 1, args.iter)
    elif args.kind == 1:
        output_test(args.path, args.gpu_id, args.saved_path, args.regression == 1, args.useimage == 1, args.usedoc == 1)
    else:
        build_imagesets(args.path)
