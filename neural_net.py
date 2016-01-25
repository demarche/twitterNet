from chainer import Link, Chain, ChainList, computational_graph, cuda, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import chainer
from chainer.functions.loss.mean_squared_error import mean_squared_error

import pickle
from sklearn import preprocessing, cross_validation
import six
import numpy as np


if __name__ == '__main__':

    # make model
    print("make model")
    n_units = 4000
    '''
    model = chainer.FunctionSet(
            l1=L.Linear(5096,  n_units),
            l2=L.Linear(n_units, 4))
    '''
    model = chainer.FunctionSet(
            l1_x1=L.Linear(1000, n_units),
            l1_x2=L.Linear(4096, n_units),
            l2_x1=L.Linear(n_units, 150),
            l2_x2=L.Linear(n_units, 150),
            l3=L.Linear(100, 4),
            l4=L.Bilinear(150, 150, 100),
    )

    cuda.get_device(0).use()
    model.to_gpu()
    xp = cuda.cupy

    def forward(x_data, y_data, train=True):
        # 入力と教師データ

        x1 = chainer.Variable(xp.asarray(x_data[:, :1000]))
        x2 = chainer.Variable(xp.asarray(x_data[:, 1000:]))
        t = chainer.Variable(xp.asarray(y_data))
        '''
        x = chainer.Variable(xp.asarray(x_data))
        t = chainer.Variable(xp.asarray(y_data))
        '''

        # 隠れ層1の出力
        '''
        h = F.softmax(model.l1(x))
        y = model.l2(h)
        '''
        h1, h2 = F.relu(model.l1_x1(x1)), F.dropout(model.l1_x2(F.relu(x2)))
        #h3 = F.relu(model.l2_x1(h1)) + F.relu(model.l2_x2(h2))
        h3 = model.l4(F.relu(model.l2_x1(h1)), F.relu(model.l2_x2(h2)))
        y = model.l3(h3)
        # 訓練時とテスト時で返す値を変える
        if train:
            # 訓練時は損失を返す
            # 多値分類なのでクロスエントロピーを使う
            loss = F.softmax_cross_entropy(y, t)
            return loss
        else:
            # テスト時は精度を返す
            acc = F.accuracy(y, t)
            return acc

    print("load feature and answer")

    features = pickle.load(open("features.dump", "rb"))
    labels = pickle.load(open("labels.dump", "rb"))
    '''
    def f(t):
        if t>=2:
            return 2
        else:
            return t
    labels = list(map(f,labels))
    # 均衡化
    '''
    lbl_cnt = []
    for i in range(max(labels)+1):
        lbl_cnt.append(labels.count(i))
    print(lbl_cnt)
    k = min(lbl_cnt)
    isUse = [False]*len(labels)
    mypm = np.random.permutation(len(labels))
    for t in range(max(labels)+1):
        counter=0
        for pm, lbl in zip(mypm, np.array(labels)[mypm]):
            if counter >= k:
                break
            if t == lbl:
                isUse[pm] = True
                counter += 1

    labels = list(map(lambda t: t[1], list(filter(lambda i: isUse[i[0]]==True, list(enumerate(labels))))))
    features = list(map(lambda t: t[1], list(filter(lambda i: isUse[i[0]]==True, list(enumerate(features))))))

    #print("scaling")
    #features = preprocessing.scale(features)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(np.array(features, dtype=np.float32),
                                                                         np.array(labels, dtype=np.int32),
                                                                         test_size=0.3, random_state=0)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    batchsize = 100
    n_epoch = 100
    N = len(y_train)
    N_test = y_test.size

    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N, batchsize):
            x_batch = x_train[perm[i:i + batchsize]]
            y_batch = y_train[perm[i:i + batchsize]]

            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.zero_grads()
            loss = forward(x_batch, y_batch)
            loss.backward()
            optimizer.update()

            if epoch == 1 and i == 0:
                with open('graph.dot', 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (loss, ), remove_split=True)
                    o.write(g.dump())
                print('graph generated')

            sum_loss += float(loss.data) * len(y_batch)

        print("train mean loss: %f" % (sum_loss / N))

        # テストデータを用いて精度を評価する
        sum_accuracy = [0.0]*(max(y_test)+1)
        for i in range(0, N_test, batchsize):
            x_batch = x_test[i:i + batchsize]
            y_batch = y_test[i:i + batchsize]

            #acc = forward(x_batch, y_batch, train=False)
            #sum_accuracy += float(acc.data) * len(y_batch)

            max_class = max(y_test)+1
            acc = [0.0]*max_class
            for i in range(max_class):
                mynumlist = [x for x in np.arange(len(y_batch)) if y_batch[x] == i]
                if len(mynumlist)==0:
                    acc[i]=0.0
                else:
                    acc[i] = forward(x_batch[mynumlist], y_batch[mynumlist], train=False).data
            sum_accuracy = [s+float(t)*len(y_batch) for s,t in zip(sum_accuracy, acc)]

        #print("test accuracy: %f" % (sum_accuracy / N_test))
        sum_accuracy = map(lambda t:t/N_test,sum_accuracy)
        print("\n".join(map(str, sum_accuracy)))

    # Save the model and the optimizer
    print('save the model')
    serializers.save_hdf5('mlp.model', model)
    print('save the optimizer')
    serializers.save_hdf5('mlp.state', optimizer)