import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class GoogLeNet(chainer.FunctionSet):

    insize = 224

    def __init__(self, n_outputs):
        super(GoogLeNet, self).__init__(
            conv1=L.Convolution2D(3,  64, 7, stride=2, pad=3),
            conv2_reduce=L.Convolution2D(64,  64, 1),
            conv2=L.Convolution2D(64, 192, 3, stride=1, pad=1),
            inc3a=L.Inception(192,  64,  96, 128, 16,  32,  32),
            inc3b=L.Inception(256, 128, 128, 192, 32,  96,  64),
            inc4a=L.Inception(480, 192,  96, 208, 16,  48,  64),
            inc4b=L.Inception(512, 160, 112, 224, 24,  64,  64),
            inc4c=L.Inception(512, 128, 128, 256, 24,  64,  64),
            inc4d=L.Inception(512, 112, 144, 288, 32,  64,  64),
            inc4e=L.Inception(528, 256, 160, 320, 32, 128, 128),
            inc5a=L.Inception(832, 256, 160, 320, 32, 128, 128),
            inc5b=L.Inception(832, 384, 192, 384, 48, 128, 128),

            loss3_fc1=L.Linear(4096, 300),
            loss3_fc2=L.Linear(300, n_outputs),

            doc_fc1=L.Linear(1000, 600),
            doc_fc2=L.Linear(600, 300),
            bi1=L.Bilinear(300, 300, 300),

            loss1_conv=L.Convolution2D(512, 128, 1),
            loss1_fc1=L.Linear(4 * 4 * 128, 1024),
            loss1_fc2=L.Linear(1024, n_outputs),

            loss2_conv=L.Convolution2D(528, 128, 1),
            loss2_fc1=L.Linear(4 * 4 * 128, 1024),

            loss2_fc2=L.Linear(1024, n_outputs),
        )

    def forward(self, x_img, x_doc, y_data, train=True):
        x_img = cuda.cupy.asarray(x_img)
        x_doc = cuda.cupy.asarray(x_doc)
        y_data = cuda.cupy.asarray(y_data)

        img, doc, t = Variable(x_img), Variable(x_doc), Variable(y_data)

        h = F.relu(self.conv1(img))
        h = F.local_response_normalization(
        F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(
        F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss1_conv(l))
        l = F.relu(self.loss1_fc1(l))
        l = self.loss1_fc2(l)
        self.loss1 = F.softmax_cross_entropy(l, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss2_conv(l))
        l = F.relu(self.loss2_fc1(l))
        l = self.loss2_fc2(l)
        self.loss2 = F.softmax_cross_entropy(l, t)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc1(F.dropout(h, 0.4, train=train))

        h2 = F.relu(self.doc_fc1(F.dropout(doc, train=train)))
        h2 = F.relu(self.doc_fc2(h2))
        b = F.relu(self.bi1(h, h2))
        h = self.loss3_fc2(b)

        self.loss3 = F.softmax_cross_entropy(h, t)

        if train:
            return 0.3 * (self.loss1 + self.loss2) + self.loss3
        else:
            return F.accuracy(h, t)
