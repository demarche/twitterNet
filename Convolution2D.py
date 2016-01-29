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
    p = ProgressBar(max_value=len(labels), min_value=1)
    for i, label in enumerate(labels):
        p.update(i+1)
        if counter[label] >= k:
            continue
        isUse[i] = True
        counter[label] += 1
    split_perm = np.array([x for x in split_perm if isUse[x]])
    p.finish()
    return split_perm

def output_test(path, gpu_id, saved_path, regression):
    gpu = gpu_id >= 0
    dir = os.path.dirname(saved_path)
    if gpu:
        cuda.check_cuda_available()
    print("make NN model..")
    worker = twitterNet_worker(1)

    print("loading NN model..")
    worker.load(saved_path)
    if gpu:
        cuda.get_device(gpu_id).use()
        worker.model.to_gpu()

    labels = np.array(pickle.load(open(os.path.join(path, 'answers_RT.pkl'), "rb")), dtype=np.int32)
    doc_vectors = np.array(pickle.load((open(os.path.join(path, "corpus_features.pkl"), "rb"))), dtype=np.float32)

    perm = pickle.load(open(os.path.join(dir, 'test_perm.pkl'), "rb"))
    batchsize = 30

    p = ProgressBar(max_value=len(perm), min_value=1)
    for i in six.moves.range(0, len(perm), batchsize):
        p.update(i+1)
        x_batch = get_images(perm[i:i + batchsize])
        x_batch_doc = doc_vectors[perm[i:i + batchsize]]
        y_batch = labels[perm[i:i + batchsize]]
        res = worker.predict(x_batch, x_batch_doc, regression, gpu=gpu)
        res = cuda.to_cpu(res.data)
        res = [[worker.fixedLog(x[0]+worker.const), worker.fixedLog(y+worker.const)] for x, y in zip(list(res), y_batch)]
        with open('some.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(res)
    p.finish()

def create_split_perm(regression, labels, path):
    label_max = max(labels) + 1
    db_len = len(labels)
    split_perm = np.random.permutation(db_len)
    if not regression:
        # 均衡化
        lbl_cnt = [labels.count(x) for x in range(label_max)]
        print(lbl_cnt)
        k = min(lbl_cnt)
        print(k)
        split_perm = reduce_label(labels, k, split_perm)
        # check
        lbl_check=[]
        for i in range(label_max):
            lbl_check.append(list(labels[split_perm]).count(i))
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
        split_perm = np.array(split_perm)

        labels2 = [labels[x] for x in split_perm]
        label_max = max(labels2) + 1
        lbl_cnt = [labels2.count(x) for x in range(label_max)]
        print(lbl_cnt)
        print(len(split_perm))
    return split_perm

def train_and_test(path, gpu_id, saved_path, regression):
    train_test_rate = 0.2
    batchsize = 10
    n_epoch = 1000
    gpu = gpu_id >= 0

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
        dir = os.path.abspath(os.path.dirname(__file__))
    else:
        dir = os.path.dirname(saved_path)
    if dir != "" and os.path.exists(os.path.join(dir, "train_perm.pkl")) and os.path.exists(os.path.join(dir, "test_perm.pkl")):
        train_perm = pickle.load(open(os.path.join(dir, "train_perm.pkl"), "rb"))
        test_perm = pickle.load(open(os.path.join(dir, "test_perm.pkl"), "rb"))
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
        if dir == "":
            dir = os.path.abspath(os.path.dirname(__file__))
        pickle.dump(train_perm, open(os.path.join(dir, "train_perm.pkl"), "wb"))
        pickle.dump(test_perm, open(os.path.join(dir, "test_perm.pkl"), "wb"))

    if regression:
        labels = np.array(labels, dtype=np.float32).reshape(len(labels), 1)
    else:
        labels = np.array(labels, dtype=np.int32)

    # make model
    print("make model")
    if regression:
        worker = twitterNet_worker(1)
    else:
        worker = twitterNet_worker(label_max)
    if saved_path != "" and os.path.exists(saved_path):
        loaded_epoch = worker.load(saved_path)
    else:
        loaded_epoch = 1
    if gpu:
        cuda.get_device(0).use()
        worker.model.to_gpu()
    train_losses = []
    train_scores = []
    test_move = []

    for epoch in six.moves.range(loaded_epoch, n_epoch + 1):
        print('epoch', epoch)
        # training
        perm = np.random.permutation(N)
        sum_loss = 0
        sum_predict = 0
        p = ProgressBar(max_value=N, min_value=1)
        for i in six.moves.range(0, N, batchsize):
            p.update(i+1)
            x_batch = get_images(train_perm[perm[i:i + batchsize]], path)
            x_batch_doc = doc_vectors[train_perm[perm[i:i + batchsize]]]
            y_batch = labels[train_perm[perm[i:i + batchsize]]]
            # Pass the loss function (Classifier defines it) and its arguments
            loss, predict = worker.train(x_batch, x_batch_doc, y_batch, regression=regression, gpu=gpu)
            sum_loss += float(loss) * len(y_batch)
            sum_predict += float(predict) * len(y_batch)
        p.finish()
        print("train mean loss: %f" % (sum_loss / N))
        print("train mean corr: %f" % (sum_predict / N))
        train_losses.append(sum_loss / N)
        train_scores.append(sum_predict / N)
        pickle.dump(train_losses, open("train_losses.pkl", "wb"))
        pickle.dump(train_scores, open("train_scores.pkl", "wb"))
        # test
        if epoch % 5 == 0 and epoch != 0:
            sum_accuracy = [0.0] * label_max
            total_acc_elem = [0] * label_max
            sum_corr = 0
            p = ProgressBar(max_value=N_test, min_value=1)
            for i in range(0, N_test, batchsize):
                p.update(i+1)
                x_batch = get_images(test_perm[i:i + batchsize], path)
                x_batch_doc = doc_vectors[test_perm[i:i + batchsize]]
                y_batch = labels[test_perm[i:i + batchsize]]

                if regression:
                    corr = worker.test(x_batch, x_batch_doc, y_batch, regression=regression, gpu=gpu)
                    sum_corr += corr * len(y_batch)
                else:
                    # ラベルごとの精度を出す
                    acc = [0.0] * label_max
                    for label in range(label_max):
                        labeled_perm = [x for x in np.arange(len(y_batch)) if y_batch[x] == label]
                        if len(labeled_perm) != 0:
                            acc = worker.test(x_batch[labeled_perm], x_batch_doc[labeled_perm], y_batch[labeled_perm],
                                             regression=regression, gpu=gpu)*len(labeled_perm)
                        total_acc_elem[label] += len(labeled_perm)
                    sum_accuracy = [s+float(t) for s, t in zip(sum_accuracy, acc)]
            p.finish()
            if regression:
                print("test mean corr: %f" % (sum_corr / N_test))
                test_move.append(sum_corr / N_test)
                pickle.dump(test_move, open("test_score.pkl", "wb"))
            else:
                sum_accuracy = [t / float(u) for t, u in zip(sum_accuracy, total_acc_elem)]
                print("\n".join(map(str, sum_accuracy)))
                print("mean:", np.mean(sum_accuracy))
                test_move.append(np.mean(sum_accuracy))
                pickle.dump(test_move, open("test_score.pkl", "wb"))

        if epoch % 5 == 0 and epoch != 0:
            # Save the model and the optimizer
            print('save the model')
            worker.save(epoch, dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process about tweetNet')
    parser.add_argument('-k', "--kind", dest='kind', default=0, type=int,
                        help='kind of process. 0: train and test. 1: test and output. 2: build image sets.  (default: 0)')
    parser.add_argument('-p', "--path", dest='path', default=os.path.abspath(os.path.dirname(__file__))+"\\data", type=str,
                        help='working path.  (default: my dir)')
    parser.add_argument('-s', "--saved_path", dest='saved_path', default="", type=str,
                        help='saved model path.  (default: empty)')
    parser.add_argument('-g', "--gpu", dest='gpu_id', default=0,
                        help='using gpu id.  (default: 0)')
    parser.add_argument('-r', "--reg", dest='regression', type=bool, default=True,
                        help='using regression predict.  (default: True)')

    args = parser.parse_args()
    if args.kind == 0:
        train_and_test(args.path, args.gpu_id, args.saved_path, args.regression)
    elif args.kind == 1:
        output_test(args.path, args.gpu_id, args.saved_path, args.regression)
    else:
        build_imagesets(args.path)

