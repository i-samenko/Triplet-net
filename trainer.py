import torch
import numpy as np
from tqdm import tqdm

from utils import plotter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from utils import accuracy_score

# def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[], start_epoch=0):

def fit(train_dataloader, val_dataloader, model, optimizer, loss_fn, n_epochs, post_classification=False,
        with_acc=False, with_scheduler=False):
    train_is_triplet = train_dataloader.dataset.is_triplet
    val_is_triplet = val_dataloader.dataset.is_triplet
    train_len = len(train_dataloader)
    val_len = len(val_dataloader)
    history = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        for inputs, labels in tqdm(train_dataloader, desc="Train iteration"):
            if with_scheduler:
                prev_sd = model.state_dict()

            optimizer.zero_grad()
            if train_is_triplet:
                anchor, pos, neg = inputs
                anchor = anchor.to(device='cuda')
                pos = pos.to(device='cuda')
                neg = neg.to(device='cuda')
                outputs = model(anchor, pos, neg)
                if not post_classification:
                    loss = loss_fn(*outputs)
                elif post_classification:
                    y0 = torch.tensor([0 for _ in range(len(outputs[0]))]).to(device='cuda')
                    y1 = torch.tensor([1 for _ in range(len(outputs[1]))]).to(device='cuda')
                    y = torch.cat((y0, y1))
                    outputs = torch.cat((outputs[0], outputs[1]))
                    loss = loss_fn(outputs, y)

            elif not train_is_triplet:
                w0, w1 = inputs
                w0 = w0.to(device='cuda')
                w1 = w1.to(device='cuda')
                labels = labels.to(device='cuda')

                outputs = model(w0, w1)
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            if with_acc:
                if post_classification:
                    train_acc += accuracy_score(outputs, y)
                elif not post_classification:
                    train_acc += accuracy_score(outputs, labels)

            train_loss += loss.item()
        model.eval();

        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc="Val iteration"):
                if val_is_triplet:
                    anchor, pos, neg = inputs
                    anchor = anchor.to(device='cuda')
                    pos = pos.to(device='cuda')
                    neg = neg.to(device='cuda')
                    outputs = model(anchor, pos, neg)
                    loss = loss_fn(*outputs)
                else:
                    w0, w1 = inputs
                    w0 = w0.to(device='cuda')
                    w1 = w1.to(device='cuda')
                    labels = labels.to(device='cuda')
                    if not post_classification:
                        outputs = model(w0, w1)
                    elif post_classification:
                        outputs = model.classifire_it(w0, w1)
                    loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                if with_acc:
                    val_acc += accuracy_score(outputs, labels)

        epoch_train_loss = train_loss / train_len
        epoch_val_loss = val_loss / val_len

        if with_acc:
            epoch_train_acc = train_acc / train_len
            epoch_val_acc = val_acc / val_len


        if with_scheduler:
            flag = False
            if epoch > 2:
                flag = True
                if epoch_val_loss > prev_loss:
                    model.load_state_dict(prev_sd)
                    epoch_val_loss = prev_loss
                    optimizer.param_groups[0]['lr'] /= 1.5
                    flag = False
            if (epoch <= 2) or (flag == True):
                history.append([epoch, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc])
                flag = False
            prev_loss = epoch_val_loss

        elif not with_scheduler:
            if with_acc:
                history.append([epoch, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc])
            else:
                history.append([epoch, epoch_train_loss, epoch_val_loss])

        plotter(history)


def get_distance_distrib(model, dev_dataloader):
    dist = []
    y = []
    model.eval();
    with torch.no_grad():
        for inp, tar in tqdm(dev_dataloader, desc="Dist iteration"):
            w0, w1 = inp
            y.append(tar.cpu().data.numpy())
            w0_emb = model.get_embedding(w0.to(device='cuda')).data.cpu().numpy()
            w1_emb = model.get_embedding(w1.to(device='cuda')).data.cpu().numpy()
            for a, b in zip(w0_emb, w1_emb):
                dist.append(cosine_distances([a], [b])[0][0])
    dist = np.array(dist)
    y = np.concatenate(y)
    return dist, y


def plot_dev_destrib(model, dev_dataloader):
    distrib,y = get_distance_distrib(model=model, dev_dataloader=dev_dataloader)
    #y = np.concatenate([tar.cpu().data.numpy() for inp, tar in dev_dataloader])
    plt.hist(distrib[y == 0], label='synonyms')
    plt.hist(distrib[y == 1], label='antonyms')
    plt.legend()
    plt.show();


def get_embedding_for_test_task(model, dataset):
    dist_f = []
    dist_t = []
    y = []
    triplet_data = []
    fasttest_data = []
    triplet_fattext_data = []

    for inputs, labels in tqdm(dataset):
        w0 = inputs[0]
        w1 = inputs[1]
        # dist_f.append(cosine_distances([w0],[w1])[0][0])
        mw0 = model.get_embedding(torch.Tensor(w0).to(device='cuda')).cpu().data.numpy()
        mw1 = model.get_embedding(torch.Tensor(w1).to(device='cuda')).cpu().data.numpy()
        # dist_t.append(cosine_distances([mw0],[mw1])[0][0])
        fasttest_data.append(np.hstack((w0, w1)))
        triplet_data.append(np.hstack((mw0, mw1)))
        triplet_fattext_data.append(np.hstack((np.hstack((w0, mw0)), np.hstack((w1, mw1)))))
        y.append(labels)

    y = np.array(y)
    fasttest_data = np.array(fasttest_data)
    triplet_data = np.array(triplet_data)
    triplet_fattext_data = np.array(triplet_fattext_data)

    return fasttest_data, triplet_data, triplet_fattext_data, y



