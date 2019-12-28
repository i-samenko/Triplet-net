import torch.nn as nn
import torch.nn.functional as F
import torch

class ClassificationNet(nn.Module):
    #https://github.com/ec2604/Antonym-Detection/blob/master/model_creation.py
    #input(300x2) -> Dence(10)->DropOut(0.1) -> Dence(5)->DropOut(0.1) ->BN()->DropOut(0.1)->Flatten()->Dence(2)

    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(600, 10)
        self.fc2 = nn.Linear(10, 5)
        self.bn1 = nn.BatchNorm1d(5)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.bn1(x)
        x = F.dropout(x,  p=0.1, training=self.training)
        #x = x.view(-1,10)
        x = self.fc3(x)
        return x #F.log_softmax(x,dim=1)
    def get_embedding(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        #print(anchor.shape, positive.shape, negative.shape)

        output1 = self.embedding_net(anchor)
        output2 = self.embedding_net(positive)
        output3 = self.embedding_net(negative)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(300, 100),
                                    nn.PReLU(),
                                    nn.Linear(100, 50))

        #self.fc1 = nn.Linear(300, 100)
        #self.fc2 = nn.Linear(100, 50)

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = nn.PReLU(x)
        # output = self.fc2(x)
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletCosLoss(nn.Module):
    def __init__(self, margin):
        super(TripletCosLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        sim = torch.nn.CosineSimilarity()
        distance_positive = 1-sim(anchor, positive)
        distance_negative = 1-sim(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

        self.fc1 = nn.Linear(100, 25)
        self.bn1 = nn.BatchNorm1d(25)
        self.fc2 = nn.Linear(25, 2)

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        concat = torch.cat((output1, output2), 1)
        # print(output1.shape, output2.shape, concat.shape)
        x = F.relu(self.fc1(concat))
        x = self.bn1(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet_v2(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet_v2, self).__init__()
        self.embedding_net = embedding_net

        self.fc1 = nn.Linear(100, 25)
        self.bn1 = nn.BatchNorm1d(25)
        self.fc12 = nn.Linear(25, 2)

        self.fc2 = nn.Linear(100, 25)
        self.bn2 = nn.BatchNorm1d(25)
        self.fc22 = nn.Linear(25, 2)

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        concat1 = torch.cat((output1, output2), 1)
        concat2 = torch.cat((output1, output3), 1)
        return concat1, concat2
        # print(output1.shape, output2.shape, concat.shape)
        # x1 = F.relu(self.fc1(concat1))
        # x1 = self.bn1(x1)
        # x1 = self.fc12(x1)

        # x2 = F.relu(self.fc2(concat2))
        # x2 = self.bn2(x2)
        # x2 = self.fc22(x2)
        # return x1, x2

    def get_embedding(self, x):
        return self.embedding_net(x)


class ClassificationeNet_v2(nn.Module): #for TripletNet_v2
    def __init__(self):
        super(ClassificationeNet_v2, self).__init__()
        self.fc = nn.Sequential(nn.Linear(100, 50),
                                nn.PReLU(),
                                nn.Linear(50, 2))

    def forward(self, x):
        output = self.fc(x)
        return output


class TripletNet_v3(nn.Module):
    def __init__(self, embedding_net, classification_net):
        super(TripletNet_v3, self).__init__()
        self.embedding_net = embedding_net
        self.classification_net = classification_net

        self.fc1 = nn.Linear(100, 25)
        self.bn1 = nn.BatchNorm1d(25)
        self.fc12 = nn.Linear(25, 2)

        self.fc2 = nn.Linear(100, 25)
        self.bn2 = nn.BatchNorm1d(25)
        self.fc22 = nn.Linear(25, 2)

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        concat1 = torch.cat((output1, output2), 1)
        concat2 = torch.cat((output1, output3), 1)
        # return concat1, concat2
        c_out1 = self.classification_net(concat1)
        c_out2 = self.classification_net(concat2)
        return c_out1, c_out2

        # print(output1.shape, output2.shape, concat.shape)
        # x1 = F.relu(self.fc1(concat1))
        # x1 = self.bn1(x1)
        # x1 = self.fc12(x1)

        # x2 = F.relu(self.fc2(concat2))
        # x2 = self.bn2(x2)
        # x2 = self.fc22(x2)
        # return x1, x2

    def get_embedding(self, x):
        return self.embedding_net(x)

    def classifire_it(self, w0, w1):
        w0_emb = self.get_embedding(w0)
        w1_emb = self.get_embedding(w1)
        concat = torch.cat((w0_emb, w1_emb), 1)
        return self.classification_net(concat)


class ClassificationeNet_v3(nn.Module):
    def __init__(self):
        super(ClassificationeNet_v3, self).__init__()
        self.fc = nn.Sequential(nn.Linear(100, 50),
                                nn.PReLU(),
                                nn.Linear(50, 2))

    def forward(self, x):
        output = self.fc(x)
        return output



