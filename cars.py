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
import losswise

from PIL import Image
from tqdm import tqdm

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# plt.ion()  # interactive mode


class CarsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mat_anno, root_dir, car_names, cleaned=None, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            root_dir (string): Directory with all the images.
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

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.car_annotations[idx][-1][0])
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


def main(key):
    num_epochs = 120  # into json file
    cars_data = CarsDataset('../../../data/cars/devkit/cars_train_annos.mat',
                            '../../../data/cars/cars_train',
                            '../../../data/cars/devkit/cars_meta.mat',
                            cleaned='../../../data/cars/cleaned.dat',
                            transform=transforms.Compose([
                                transforms.Scale(350),
                                transforms.RandomSizedCrop(270),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
                            )

    cars_data_test = CarsDataset('../../../data/cars/devkit/cars_train_annos.mat',
                            '../../../data/cars/cars_train',
                            '../../../data/cars/devkit/cars_meta.mat',
                            cleaned='../../../data/cars/cleaned_test.dat',
                            transform=transforms.Compose([
                                transforms.Scale(270),
                                # transforms.RandomSizedCrop(270),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
                            )

    dataloader = DataLoader(cars_data, batch_size=8,
                            shuffle=True, num_workers=8)
    print("Train data set length:", len(cars_data))

    testloader = DataLoader(cars_data_test, batch_size=4,
                            shuffle=True, num_workers=4)
    print("Test data set length:", len(cars_data_test))

    losswise.set_api_key(key)


    # for i, batch in enumerate(dataloader):
    #     plt.figure(figsize=(15,5))
    #     img, classes = batch
    #     cars_data.show_batch(img.numpy(), classes.numpy())
    #
    #     plt.show()
    #     plt.pause(4)
    #     plt.clf()

    session = losswise.Session(tag='Resnet18_cars', max_iter=num_epochs)
    graph_tloss = session.graph('loss', kind='min')
    graph_acc = session.graph('accuracy', kind='max')


    model_ft = models.resnet18(pretrained=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 197)

    model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=12, gamma=0.1)

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

            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            optimizer_ft.zero_grad()

            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()

            running_loss += loss.data[0]
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            # if c % 500 == 499:
            #     print('Batchloss: {:.4f}'.format(running_loss / c))

        epoch_loss = running_loss / c
        train_acc = 100 * correct / total
        print('Train epoch: {} || Loss: {:.4f} || Acc: {:.2f} %%'.format(
            epoch, epoch_loss, train_acc))

        graph_tloss.append(epoch, {'train_loss': running_loss / c})
        graph_acc.append(epoch, {'train_acc': train_acc})



        correct = 0
        total = 0
        for data in tqdm(testloader):
            images, labels = data
            labels = labels.type(torch.LongTensor).cuda()
            images = Variable(images).cuda()
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        test_acc = 100 * correct / total
        print('Test Acc: {:.2f} %%'.format(test_acc))

        graph_acc.append(epoch, {'test_acc': test_acc})

    session.done()


if __name__ == '__main__':
    main(sys.argv[1])




# if np_input.shape[-3] != 3:
# plt.figure(figsize=(15, 5))
# print("num++++++++++++++++++++++++++++++++++++++", np_input.shape)
# cars_data.show_batch(inputs.numpy(), labels.numpy())
# plt.show()
# plt.pause(1)
# plt.clf()
