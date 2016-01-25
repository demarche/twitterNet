from chainer import Link, Chain, ChainList, computational_graph, cuda, optimizers, serializers, FunctionSet, Variable
import chainer.functions as F
import chainer.links as L

in_size = 256
path = r"C:\Users\大悟\Desktop\moved\moved\images"

class RetweetNet(FunctionSet):
    def __init__(self, n_outputs):
        super(RetweetNet, self).__init__(
            conv1=L.Convolution2D(3, 96, ksize=11, stride=4, pad=0),
            conv2=L.Convolution2D(96, 256, ksize=5, pad=2),
            conv3=L.Convolution2D(256, 384, ksize=3, pad=1),
            conv4=L.Convolution2D(384, 384, ksize=3, pad=1),
            conv5=L.Convolution2D(384, 256, ksize=3, pad=1),
            fc6=L.Linear(12544, 4096),
            fc7=L.Linear(4096, 300),
            doc_fc1=L.Linear(1000, 600),
            doc_fc2=L.Linear(600, 300),
            bi1=L.Bilinear(300, 300, 200),
            fc8=L.Linear(200, n_outputs),
        )

    def forward(self, x_img, x_doc, y_data, train=True):

        x_img = cuda.cupy.asarray(x_img)
        x_doc = cuda.cupy.asarray(x_doc)
        y_data = cuda.cupy.asarray(y_data)

        img, doc, t = Variable(x_img), Variable(x_doc), Variable(y_data)

        h = F.max_pooling_2d(F.relu(self.conv1(img)), ksize=3, stride=2, pad=0)
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=2, pad=0)
        h = F.local_response_normalization(h)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), ksize=3, stride=2, pad=0)
        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
        h2 = F.relu(self.doc_fc1(doc))
        h2 = F.relu(self.doc_fc2(h2))
        b = F.relu(self.bi1(h, h2))
        y = self.fc8(b)
        if train:
            return F.softmax_cross_entropy(y, t)
        else:
            return F.accuracy(y, t)