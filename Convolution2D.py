from chainer import Link, Chain, ChainList, computational_graph, cuda, optimizers, serializers, FunctionSet, Variable
from RetweetNet_Org import RetweetNet
from GoogleNet import GoogLeNet
from GoogleNetBN import GoogLeNetBN

import pickle
from sklearn import preprocessing, cross_validation
import six, sys
import numpy as np
import csv
import glob
from PIL import Image
from progressbar import ProgressBar
import pickle
import os.path
from extract_feature import saver

in_size = 256


def get_images(perm):

    mean_image = pickle.load(open("image_mean.pkl", "rb"))
    img_list=[]
    for img_name in perm:
        img = pickle.load(open("/media/yamashita004/4dad8012-5855-4d11-8128-8fc5247ba677/NeuralNet/images/"+str(img_name)+".pkl", "rb"))
        img -= mean_image
        img_list.append(img)
    return np.array(img_list, dtype=np.float32)

def build_imagesets():
    my_saver = saver(r'/media/yamashita004/HDPX-UT/moved', in_size)
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

if __name__ == '__main2__':
    build_imagesets()

if __name__ == "__main2__":
    if len(sys.argv)>1:
        gpu = sys.argv[1]=="1"
    else:
        gpu=False
    if gpu:cuda.check_cuda_available()
    print("make NN model..")

    optimizer = optimizers.Adam()
    model = GoogLeNetBN(1)

    print("loading NN model..")
    serializers.load_hdf5(r'40mlp.model', model)
    if gpu:
        cuda.get_device(0).use()
        model.to_gpu()

    #my_saver = saver(r'/media/yamashita004/HDPX-UT/moved', in_size)

    labels = np.array(pickle.load(open('answers_RT.pkl', "rb")), dtype=np.int32)
    doc_vectors = np.array(pickle.load((open("corpus_features.pkl", "rb"))), dtype=np.float32)

    perm = pickle.load(open('test_perm.pkl', "rb"))
    batchsize = 20

    p = ProgressBar(max_value=len(perm), min_value=1)
    for i in six.moves.range(0, len(perm), batchsize):
        p.update(i+1)
        x_batch = get_images(perm[i:i + batchsize])
        x_batch_doc = doc_vectors[perm[i:i + batchsize]]
        y_batch = labels[perm[i:i + batchsize]]
        res = model.forward(x_batch, x_batch_doc, y_batch, train=False, regression=True, gpu=gpu)
        res = cuda.to_cpu(res.data)
        res = [[model.fixedLog(x[0]+model.const), model.fixedLog(y+model.const)] for x, y in zip(list(res), y_batch)]
        with open('some.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(res)
    p.finish()

if __name__ == '__main__':
    train_test_rate = 0.2
    batchsize = 30
    n_epoch = 1000
    regression = False

    print("loading data")
    if regression:
        labels = pickle.load(open(
                '/media/yamashita004/4dad8012-5855-4d11-8128-8fc5247ba677/NeuralNet/answers_RT.pkl', "rb"))
    else:
        labels = pickle.load(open("answers.pkl", "rb"))
    labels = list(map(int, labels))
    lbmax = max(labels)+1
    doc_vectors = pickle.load((open("corpus_features.pkl", "rb")))
    db_len = len(labels)
    split_perm = np.random.permutation(db_len)

    print("normalarize")
    if not regression:
        # 均衡化
        lbl_cnt = [labels.count(x) for x in range(lbmax)]
        print(lbl_cnt)
        k = min(lbl_cnt)
        print(k)

        split_perm = reduce_label(labels, k, split_perm)

        labels = np.array(labels, dtype=np.int32)
    else:
        if os.path.exists("lbl_cnt.pkl"):
            print("load lbl_cnt")
            lbl_cnt = pickle.load(open("lbl_cnt.pkl", "rb"))
        else:
            print("create lbl_cnt")
            lbl_cnt = []
            p = ProgressBar(max_value=lbmax, min_value=1)
            for i, x in enumerate(range(lbmax)):
                p.update(i+1)
                lbl_cnt.append(labels.count(x))
            p.finish()
            pickle.dump(lbl_cnt, open("lbl_cnt.pkl", "wb"))

        print("create mean")
        #lbl_cnt_mean = np.mean(list(filter(lambda t:t>10, lbl_cnt)))
        lbl_cnt_mean=500
        print(lbl_cnt)
        print(lbl_cnt_mean)

        # reduce
        split_perm = reduce_label(labels, lbl_cnt_mean, split_perm)
        labels2 = [labels[x] for x in split_perm]
        lbl_cnt = [labels2.count(x) for x in range(lbmax)]
        print(lbl_cnt)

        # remove sparse element
        remove_list = []
        for i, x in list(enumerate(lbl_cnt))[::-1]:
            if x != 1 and x != 0:
                break
            elif i+1 == lbmax or x == 1:
                remove_list.append(i)
            elif x == 1 and (lbl_cnt[i+1] !=0 or lbl_cnt[i-1] != 0):
                break
        split_perm = list(split_perm)
        for x in remove_list:
            index = labels.index(x)
            split_perm.remove(index)
        split_perm = np.array(split_perm)

        labels2 = [labels[x] for x in split_perm]
        lbmax = max(labels2)+1
        lbl_cnt = [labels2.count(x) for x in range(lbmax)]
        print(lbl_cnt)
        print(len(split_perm))

        labels = np.array(labels, dtype=np.float32)
        labels = labels.astype(np.float32).reshape(len(labels), 1)

    doc_vectors = np.array(doc_vectors, dtype=np.float32)

    if not regression:
        # check
        lbl_check=[]
        for i in range(lbmax):
            lbl_check.append(list(labels[split_perm]).count(i))
        print(lbl_check)

    # split train test data
    db_len=len(split_perm)
    N_test = int(train_test_rate*db_len)
    N = int(db_len-N_test)
    train_perm = split_perm[:N]
    test_perm = split_perm[N:]
    print("train:", len(train_perm))
    print("test:", len(test_perm))
    pickle.dump(train_perm, open("train_perm.pkl", "wb"))
    pickle.dump(test_perm, open("test_perm.pkl", "wb"))

    # make model
    print("make model")
    #model = RetweetNet(max(labels)+1)
    if regression:
        model = GoogLeNetBN(1)
    else:
        model = GoogLeNetBN(lbmax)
    cuda.get_device(0).use()
    model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    loss_move=[]
    acc_move=[]
    test_move=[]

    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)

        # random select optimizer
        rnd = np.random.randint(0, 3)
        if rnd == 0:
            print('opt:Adam')
            optimizer = optimizers.Adam()
        elif rnd == 1:
            print('opt:AdaGrad')
            optimizer = optimizers.AdaGrad()
        else:
            print('opt:AdaDelta')
            optimizer = optimizers.AdaDelta()
        optimizer.setup(model)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        sum_corr = 0
        p = ProgressBar(max_value=N, min_value=1)
        for i in six.moves.range(0, N, batchsize):
            p.update(i+1)
            x_batch = get_images(train_perm[perm[i:i + batchsize]])
            x_batch_doc = doc_vectors[train_perm[perm[i:i + batchsize]]]
            y_batch = labels[train_perm[perm[i:i + batchsize]]]

            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.zero_grads()
            if regression:
                loss, corr = model.forward(x_batch, x_batch_doc, y_batch, regression=regression)
            else:
                loss = model.forward(x_batch, x_batch_doc, y_batch, regression=regression)
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(y_batch)
            if regression:
                sum_corr += corr[0, 1] * len(y_batch)
        p.finish()
        print("train mean loss: %f" % (sum_loss / N))
        loss_move.append(sum_loss / N)
        pickle.dump(loss_move, open("loss.pkl", "wb"))
        if regression:
            print("train mean corr: %f" % (sum_corr / N))
            acc_move.append(sum_corr / N)
            pickle.dump(acc_move, open("corr.pkl", "wb"))


        # テストデータを用いて精度を評価する
        if epoch%5 == 0 and epoch != 0:
            sum_accuracy = [0.0]*lbmax
            total_acc_elem = [0]*lbmax
            sum_loss = 0.0
            sum_corr = 0
            p = ProgressBar(max_value=N_test, min_value=1)
            for i in range(0, N_test, batchsize):
                p.update(i+1)
                x_batch = get_images(test_perm[i:i + batchsize])
                x_batch_doc = doc_vectors[test_perm[i:i + batchsize]]
                y_batch = labels[test_perm[i:i + batchsize]]

                if regression:
                    loss, corr = model.forward(x_batch, x_batch_doc, y_batch,
                                                   train=False, regression=regression)
                    sum_corr += corr[0, 1] * len(y_batch)
                else:
                    acc = [0.0]*lbmax
                    for i in range(lbmax):
                        mynumlist = [x for x in np.arange(len(y_batch)) if y_batch[x] == i]
                        if len(mynumlist)==0:
                            acc[i]=0.0
                        else:
                            acc[i] = model.forward(x_batch[mynumlist], x_batch_doc[mynumlist], y_batch[mynumlist],
                                                   train=False, regression=regression).data*len(mynumlist)
                        total_acc_elem[i] += len(mynumlist)
                    sum_accuracy = [s+float(t) for s, t in zip(sum_accuracy, acc)]
            p.finish()
            if regression:
                print("test mean corr: %f" % (sum_corr / N_test))
                test_move.append(sum_corr / N_test)
                pickle.dump(test_move, open("test_corr.pkl", "wb"))
            else:
                sum_accuracy = [t / float(u) for t, u in zip(sum_accuracy, total_acc_elem)]
                acc_move.append(np.mean(sum_accuracy))
                pickle.dump(acc_move, open("acc.pkl", "wb"))
                print("\n".join(map(str, sum_accuracy)))
                print("mean:", np.mean(sum_accuracy))

        if epoch % 5 == 0 and epoch != 0:
            # Save the model and the optimizer
            print('save the model')
            serializers.save_hdf5(str(epoch)+'mlp.model', model)
            print('save the optimizer')
            serializers.save_hdf5(str(epoch)+'mlp.state', optimizer)
