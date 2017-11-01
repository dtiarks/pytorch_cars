from __future__ import print_function, division
import os
import sys
import torch
# from skimage import io, transform
import numpy as np
import scipy.io
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import losswise
import argparse

from PIL import Image
from tqdm import tqdm

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# plt.ion()  # interactive mode


class CarsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mat_anno, data_dir, car_names, cleaned=None, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        if cleaned is not None:
            cleaned_annos = []
            print("Cleaning up data set (only take pics with rgb chans)...")
            clean_files = np.loadtxt(cleaned, dtype=str)
            for c in self.car_annotations:
                if c[-1][0] in clean_files:
                    cleaned_annos.append(c)
            self.car_annotations = cleaned_annos

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        car_class = self.car_annotations[idx][-2][0][0]

        if self.transform:
            image = self.transform(image)

        return image, car_class

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret

    def show_batch(self, img_batch, class_batch):

        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()


class Densenet161(nn.Module):
    def __init__(self, num_classes = 197, drop_rate=0.):
        super(Densenet161,self).__init__()
        original_model = models.densenet161(pretrained=True, drop_rate=drop_rate)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = (nn.Linear(2208, num_classes))

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        y = self.classifier(f)
        return y


def save_model(epoch, net, optim, ckpt_fname):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)

def load_model():
    pass
    # if pretrained is not None:
    #     state_dict = torch.load(pretrained)
    #     new_state_dict = OrderedDict()
    #     for k, value in state_dict['state_dict'].iteritems():
    #         key = "module.{}".format(k)
    #         new_state_dict[key] = value
    #     net.load_state_dict(new_state_dict)
    #     epoch = state_dict['epoch']
    #     print
    #     "pre-trained epoch number: {}".format(epoch)
    #     optimizer = state_dict['optimizer']

def visualize_batch(batch):
    pass

def train(params):
    num_epochs = params["epochs"] # into json file
    train_batch_size = params["train_batchsize"]
    train_workers = 8
    test_batch_size = 8
    test_workers = 8
    num_classes = 197
    data_dir = params["data_dir"]
    checkpoint_dir = "./checkpoint"
    save_freq = params["save_freq"]
    enable_cuda = params["enable_cuda"]
    learning_rate = params["learning_rate"]
    lr_updates = params["lr_updates"]
    lr_gamma = params["lr_gamma"]
    weight_decay = params["weight_decay"]

    cars_data = CarsDataset(os.path.join(data_dir,'devkit/cars_train_annos.mat'),
                            os.path.join(data_dir,'cars_train'),
                            os.path.join(data_dir,'devkit/cars_meta.mat'),
                            cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transforms.Compose([
                                transforms.Scale(250),
                                transforms.RandomSizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                            ])
                            )

    cars_data_val = CarsDataset(os.path.join(data_dir,'devkit/cars_test_annos_withlabels.mat'),
                            os.path.join(data_dir,'cars_test'),
                            os.path.join(data_dir,'devkit/cars_meta.mat'),
                            cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transforms.Compose([
                                transforms.Scale(224),
                                transforms.RandomSizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.46905602, 0.45872932, 0.4539325), (0.26603131, 0.26460057, 0.26935185))
                            ])
                            )

    dataloader = DataLoader(cars_data, batch_size=train_batch_size,
                            shuffle=True, num_workers=train_workers)
    print("Train data set length:", len(cars_data))

    valloader = DataLoader(cars_data_val, batch_size=test_batch_size,
                            shuffle=True, num_workers=test_workers)
    print("Validation data set length:", len(cars_data_val))

    losswise.set_api_key(params["key"])


    # for i, batch in enumerate(dataloader):
    #     plt.figure(figsize=(15,5))
    #     img, classes = batch
    #     cars_data.show_batch(img.numpy(), classes.numpy())
    #
    #     plt.show()
    #     plt.pause(4)
    #     plt.clf()

    session = losswise.Session(tag='Densnet161_cars', max_iter=num_epochs)
    graph_tloss = session.graph('loss', kind='min')
    graph_acc = session.graph('accuracy', kind='max')


    model_ft = models.resnet18(pretrained=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_val = models.resnet18(pretrained=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False
    num_val = model_val.fc.in_features
    model_val.fc = nn.Linear(num_val, num_classes)

    # model_ft = Densenet161(drop_rate=0.5)
    # model_val = Densenet161()

    if enable_cuda:
        model_ft = model_ft.cuda()
        model_val = model_val.cuda()

        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=lr_updates, gamma=lr_gamma)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        exp_lr_scheduler.step()
        model_ft.train(True)  # Set model to training mode

        running_loss = 0.0
        c = 0
        correct = 0
        total = 0
        for batch in tqdm(dataloader):
            c += 1
            inputs, labels = batch
            labels = labels.type(torch.LongTensor)

            if enable_cuda:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer_ft.zero_grad()

            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()

            running_loss += loss.data[0]
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        epoch_loss = running_loss / c
        train_acc = 100 * correct / total
        print('Train epoch: {} || Loss: {:.4f} || Acc: {:.2f} %%'.format(
            epoch, epoch_loss, train_acc))

        graph_tloss.append(epoch, {'train_loss': running_loss / c})
        graph_acc.append(epoch, {'train_acc': train_acc})

        if epoch % save_freq == 0:
            save_model(epoch, model_ft, optimizer_ft, os.path.join(checkpoint_dir, 'model_%03d.pth' % epoch))

        model_val.train(False)
        model_val.load_state_dict(model_ft.state_dict())

        correct = 0
        total = 0
        for data in tqdm(valloader):
            images, labels = data

            if enable_cuda:
                labels = labels.type(torch.LongTensor).cuda()
                images = Variable(images).cuda()
            else:
                images = Variable(images)

            outputs = model_val(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        val_acc = 100 * correct / total
        print('Validation Acc: {:.2f} %%'.format(val_acc))

        graph_acc.append(epoch, {'val_acc': val_acc})

    session.done()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", type=str, help="Losswise API key", default='NA')
    parser.add_argument("-d", "--dir", type=str, help="Directory where the training data is stored")
    parser.add_argument("-c", "--cuda", type=bool, help="Enable CUDA", default=True)
    # parser.add_argument("-c", "--checkpoint", type=str, help="Directory of latest checkpoint.")
    parser.add_argument("-n", "--name", type=str, help="Session name in Losswise.")
    args = parser.parse_args()

    params = {
        "key": args.key,
        "data_dir": args.dir,
        "enable_cuda": args.cuda,
        "learning_rate": 0.000025,
        "weight_decay": 5e-4,
        "epochs": 150,
        "train_batchsize": 32,
        "save_freq": 10,
        "lr_updates": 20,
        "lr_gamma": 0.3
    }

    train(params)




# if np_input.shape[-3] != 3:
# plt.figure(figsize=(15, 5))
# print("num++++++++++++++++++++++++++++++++++++++", np_input.shape)
# cars_data.show_batch(inputs.numpy(), labels.numpy())
# plt.show()
# plt.pause(1)
# plt.clf()
