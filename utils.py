import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics.pairwise import cosine_distances


def convert_data_to_xy(data):
    X = []
    y = []
    r = {'S': [1, 0], 'A': [0, 1], 's': [1, 0]}
    for w0, w1, relation, e0, e1 in tqdm(data):
        X.append(np.hstack((e0, e1))) #(600,)
        #X.append(np.stack((e0,e1),axis=0)) #(2,300)
        y.append(r[relation])
    return X, y


def to_categorical(labels):
    l = torch.zeros((labels.size(0), 2), device='cuda')
    l[np.arange(labels.size(0)), labels] = 1
    return l


def num_correct_samples(outputs, labels):
    max_values, max_indices = torch.max(outputs.data, 1)
    equal = max_indices.eq(labels.data).sum().item()
    return equal


def accuracy_score(outputs, labels):
    max_vals, max_indices = torch.max(outputs, 1)
    acc = (max_indices == labels).sum().cpu().data.numpy()/max_indices.size()[0]
    return acc


def plotter(history):
    _hist = np.array(history)
    with_acc = False
    if _hist.shape[1] > 3:
        with_acc = True

    train_loss, val_loss = _hist[:, 1], _hist[:, 2]
    if with_acc:
        train_acc, val_acc = _hist[:, 3], _hist[:, 4]

    clear_output(True)
    plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    plt.title('loss')
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='vall loss')
    plt.grid()
    plt.legend()
    plt.show()

    if with_acc:
        plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        plt.title('accuracy')
        plt.plot(train_acc, label='train acc')
        plt.plot(val_acc, label='vall acc')
        plt.grid()
        plt.legend()
        plt.show()

    for elem in history:
        epoch = elem[0]
        train_loss = elem[1]
        val_loss = elem[2]
        if with_acc:
            train_acc = elem[3]
            val_acc = elem[4]
        print(f'Epoch {epoch}')
        print('Train Loss: {:.4f}'.format(train_loss),
              '| Val Loss: {:.4f}'.format(val_loss))
        if with_acc:
            print('Train Acc: {:.4f}'.format(train_acc),
                  '| Val Acc: {:.4f}'.format(val_acc))

        print()


def show_raw_distrib(data,fst_emb):
    d = np.array([cosine_distances([fst_emb[w0]], [fst_emb[w1]])[0][0] for w0,w1 in data[:,:2]])
    plt.hist(d[data[:,2]=='S'], label = 'synonyms')
    plt.hist(d[data[:,2]=='A'], label = 'antonyms')
    plt.legend()
    #plt.show()
