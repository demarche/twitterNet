import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable, optimizers, serializers
import math
import numpy as np
import os
import os.path

class twitterNet_worker():

    const = 10.0

    def __init__(self, outputdim, optimizer=None):
        if optimizer is None:
            self.optimizer = chainer.optimizers.Adam()
        else:
            self.optimizer = optimizer
        self.model = GoogLeNetBN(outputdim)
        self.optimizer.setup(self.model)
        self.myOptimizers = [optimizers.Adam(), optimizers.AdaGrad(), optimizers.AdaDelta()]

    def save(self, path, epoch):
        joined_path = os.path.join(path, "model")
        if not os.path.exists(joined_path):
            os.mkdir(joined_path)
        serializers.save_hdf5(os.path.join(joined_path, str(epoch)+'mlp.model'), self.model)
        serializers.save_hdf5(os.path.join(joined_path, str(epoch)+'mlp.state'), self.optimizer)

    def load(self, path):
        name, _ = os.path.splitext(os.path.basename(path))
        serializers.load_hdf5(path, self.model)
        return int(name[:-3])

    def fixedLog(self, a):
        if a <= 0:
            a = self.const
        res = math.log(a, self.const)
        return res

    def toLog(self, t, xp):
        t.data = xp.asarray(list(map(lambda t: self.fixedLog(t[0]+self.const), t.data)),
                            dtype=cuda.cupy.float32).reshape(t.data._shape)
        return t

    def predict(self, x_img, x_doc, regression, gpu=True, useImage=True, useDoc=True):
        xp = cuda.cupy if gpu else np
        x_img = xp.asarray(x_img)
        x_doc = xp.asarray(x_doc)
        img, doc = Variable(x_img), Variable(x_doc)
        return self.model.forward(img, doc, train=False, regression=regression, useImage=useImage, useDoc=useDoc)

    def test(self, x_img, x_doc, y_data, regression, gpu=True, useImage=True, useDoc=True):
        xp = cuda.cupy if gpu else np
        x_img = xp.asarray(x_img)
        x_doc = xp.asarray(x_doc)
        y_data = xp.asarray(y_data)
        img, doc, t = Variable(x_img), Variable(x_doc), Variable(y_data)
        y = self.model.forward(img, doc, train=False, regression=regression, useImage=useImage, useDoc=useDoc)
        if regression:
            h = self.toLog(y, xp)
            t = self.toLog(t, xp)
            h = np.array(cuda.to_cpu(h.data)).reshape((len(h)))
            t = np.array(cuda.to_cpu(t.data)).reshape((len(t)))
            return np.corrcoef(h, t)[0, 1]
        else:
            return F.accuracy(y, t).data

    def train(self, x_img, x_doc, y_data, regression, gpu=True, useImage=True, useDoc=True):
        xp = cuda.cupy if gpu else np
        x_img = xp.asarray(x_img)
        x_doc = xp.asarray(x_doc)
        y_data = xp.asarray(y_data)
        img, doc, t = Variable(x_img), Variable(x_doc), Variable(y_data)
        y = self.model.forward(img, doc, regression=regression, useImage=useImage, useDoc=useDoc)

        # calc loss
        if useImage:
            if regression:
                a = self.toLog(y["a"], xp)
                b = self.toLog(y["b"], xp)
                h = self.toLog(y["h"], xp)
                t = self.toLog(t, xp)
                self.loss1 = F.mean_squared_error(a, t)
                self.loss2 = F.mean_squared_error(b, t)
                self.loss3 = F.mean_squared_error(h, t)
            else:
                a = y["a"]
                b = y["b"]
                h = y["h"]
                self.loss1 = F.softmax_cross_entropy(a, t)
                self.loss2 = F.softmax_cross_entropy(b, t)
                self.loss3 = F.softmax_cross_entropy(h, t)
            loss = 0.3 * (self.loss1 + self.loss2) + self.loss3
        else:
            if regression:
                h = self.toLog(y, xp)
                t = self.toLog(t, xp)
                self.loss1 = F.mean_squared_error(h, t)
            else:
                h = y
                self.loss1 = F.softmax_cross_entropy(y, t)
            loss = self.loss1


        # random select optimizer
        rnd = np.random.randint(0, len(self.myOptimizers))
        self.optimizer = self.myOptimizers[rnd]
        self.optimizer.setup(self.model)
        self.optimizer.zero_grads()
        loss.backward()
        self.optimizer.update()

        if regression:
            h = np.array(cuda.to_cpu(h.data)).reshape((len(h)))
            t = np.array(cuda.to_cpu(t.data)).reshape((len(t)))
            return loss.data, np.corrcoef(h, t)[0, 1]
        else:
            return loss.data, F.accuracy(h, t).data

class GoogLeNetBN(chainer.FunctionSet):

    """New GoogLeNet of BatchNormalization version."""

    def __init__(self, n_outputs):
        super(GoogLeNetBN, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, stride=2, pad=3, nobias=True),
            norm1=L.BatchNormalization(64),
            conv2=L.Convolution2D(64, 192, 3, pad=1, nobias=True),
            norm2=L.BatchNormalization(192),
            inc3a=L.InceptionBN(192, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=L.InceptionBN(256, 64, 64, 96, 64, 96, 'avg', 64),
            inc3c=L.InceptionBN(320, 0, 128, 160, 64, 96, 'max', stride=2),
            inc4a=L.InceptionBN(576, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=L.InceptionBN(576, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=L.InceptionBN(576, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=L.InceptionBN(576, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=L.InceptionBN(576, 0, 128, 192, 192, 256, 'max', stride=2),
            inc5a=L.InceptionBN(1024, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=L.InceptionBN(1024, 352, 192, 320, 192, 224, 'max', 128),
            linz=L.Linear(1024, 1024),
            out=L.Linear(2024, n_outputs),
            outimg=L.Linear(1024, n_outputs),
            outdoc=L.Linear(1000, n_outputs),

            doc_fc1=L.Linear(1000, 1000),

            conva=L.Convolution2D(576, 128, 1, nobias=True),
            norma=L.BatchNormalization(128),
            lina=L.Linear(3200, 1024, nobias=True),
            norma2=L.BatchNormalization(1024),
            outa=L.Linear(1024, n_outputs),

            convb=L.Convolution2D(576, 128, 1, nobias=True),
            normb=L.BatchNormalization(128),
            linb=L.Linear(3200, 1024, nobias=True),
            normb2=L.BatchNormalization(1024),
            outb=L.Linear(1024, n_outputs)
        )

    def forward(self, img, doc, train=True, regression=False, useImage=True, useDoc=True):
        test = not train

        if useImage:
            h = F.max_pooling_2d(
                F.relu(self.norm1(self.conv1(img), test=test)),  3, stride=2, pad=1)
            h = F.max_pooling_2d(
                F.relu(self.norm2(self.conv2(h), test=test)), 3, stride=2, pad=1)

            h = self.inc3a(h)
            h = self.inc3b(h)
            h = self.inc3c(h)
            h = self.inc4a(h)

            if train:
                a = F.average_pooling_2d(h, 5, stride=3)
                a = F.relu(self.norma(self.conva(a), test=test))
                a = F.relu(self.norma2(self.lina(a), test=test))
                a = self.outa(a)

            h = self.inc4b(h)
            h = self.inc4c(h)
            h = self.inc4d(h)

            if train:
                b = F.average_pooling_2d(h, 5, stride=3)
                b = F.relu(self.normb(self.convb(b), test=test))
                b = F.relu(self.normb2(self.linb(b), test=test))
                b = self.outb(b)

            h = self.inc4e(h)
            h = self.inc5a(h)
            h = F.average_pooling_2d(self.inc5b(h), 7)
            h = F.relu(self.linz(h))

        if useDoc:
            h2 = F.leaky_relu(self.doc_fc1(F.dropout(doc, train=train)))

        if useDoc and useImage:
            bi = F.concat((h, h2), axis=1)
            h = self.out(bi)
        elif useImage:
            h = self.outimg(h)
        else:
            h = self.outdoc(h2)

        if train:
            if useImage:
                return {
                    "a": a,
                    "b": b,
                    "h": h
                }
            else:
                return h

        else:
            return h
