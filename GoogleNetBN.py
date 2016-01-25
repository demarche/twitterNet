import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
import math
import numpy as np

class GoogLeNetBN(chainer.FunctionSet):

    """New GoogLeNet of BatchNormalization version."""

    insize = 224
    const = 10.0

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
            linz=L.Linear(1024, 300),
            out=L.Linear(300, n_outputs),

            doc_fc1=L.Linear(1000, 600),
            doc_fc2=L.Linear(600, 300),
            bi1=L.Bilinear(300, 300, 300),

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
    def fixedLog(self, a):
        if a<=0:
            a=self.const
        res = math.log(a, self.const)
        return res

    def toLogCPU(self, t):
        t.data = np.asarray(list(map(lambda t: self.fixedLog(t[0]+self.const), t.data)),
                                   dtype=np.float32).reshape(t.data.shape)
        return t

    def toLog(self, t):
        t.data = cuda.cupy.asarray(list(map(lambda t: self.fixedLog(t[0]+self.const), t.data)),
                                   dtype=cuda.cupy.float32).reshape(t.data._shape)
        return t


    def forward(self, x_img, x_doc, y_data, train=True, regression=False, predict=False, gpu=True):
        test = not train

        xp = cuda.cupy if gpu else np
        x_img = xp.asarray(x_img)
        x_doc = xp.asarray(x_doc)
        y_data = xp.asarray(y_data)

        img, doc, t = Variable(x_img), Variable(x_doc), Variable(y_data)

        if regression and not predict:
            t = self.toLog(t)
            #t.data = cuda.cupy.asarray(t.data,  dtype=cuda.cupy.float32).reshape((20,1))

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(img), test=test)),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h), test=test)), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        if not predict:
            a = F.average_pooling_2d(h, 5, stride=3)
            a = F.relu(self.norma(self.conva(a), test=test))
            a = F.relu(self.norma2(self.lina(a), test=test))
            a = self.outa(a)
            if regression:
                #a = self.toLog(a)
                self.loss1 = F.mean_squared_error(a, t)
            else:
                self.loss1 = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        if not predict:
            b = F.average_pooling_2d(h, 5, stride=3)
            b = F.relu(self.normb(self.convb(b), test=test))
            b = F.relu(self.normb2(self.linb(b), test=test))
            b = self.outb(b)
            if regression:
                #b = self.toLog(b)
                self.loss2 = F.mean_squared_error(b, t)
            else:
                self.loss2 = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = F.relu(self.linz(h))

        h2 = F.relu(self.doc_fc1(F.dropout(doc, train=train)))
        h2 = F.relu(self.doc_fc2(h2))
        b = F.relu(self.bi1(h, h2))

        h = self.out(b)

        if predict:
            #t.data = cuda.cupy.asarray(t.data,  dtype=cuda.cupy.float32).reshape((20,1))
            #myloss = F.mean_squared_error(h, t)
            return h
        if regression:
            h = self.toLog(h)
            self.loss3 = F.mean_squared_error(h, t)
        else:
            self.loss3 = F.softmax_cross_entropy(h, t)

        if train:
            if regression:
                h = np.array(cuda.to_cpu(h.data)).reshape((len(h)))
                t = np.array(cuda.to_cpu(t.data)).reshape((len(t)))
                return 0.3 * (self.loss1 + self.loss2) + self.loss3, np.corrcoef(h, t)
            else:
                return 0.3 * (self.loss1 + self.loss2) + self.loss3
        else:
            return F.accuracy(h, t)
