import torch as th
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
import numpy as np
import gzip
from matplotlib import pyplot as plt

def preprocess(inputs_path, labels_path, num_examples):
    '''
    :param inputs_path: path to data inputs (images of size [28 x 28])
    :param labels_path: path to the labels of the images
    :param num_examples: number of examples to read. 60000 for training; 10000 for testing. 
    :return: numpy array of images, size [num_examples x 784], and labels, size [num_examples]
    '''
    with open(inputs_path, "rb") as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        b1 = bytestream.read(784 * num_examples)
        inp = np.frombuffer(b1, dtype=np.uint8, count=-1).reshape(num_examples, 784)
        inp = inp.astype(np.float32)
    with open(labels_path, "rb") as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        b2 = bytestream.read(num_examples)
        labels = np.frombuffer(b2, dtype=np.uint8, count=-1)
    normalized_inputs = inp / 255
    return normalized_inputs, labels

class Model(th.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Hyperparameters Initialization
        self.batch_size = 50
        self.learning_rate = 0.001
        self.ff_layer1 = th.nn.Linear(784, 120)
        self.ff_layer2 = th.nn.Linear(120, 10)
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_layer = th.nn.CrossEntropyLoss()
    
    def call(self, inputs):
        '''
        :param inputs: batched input of images, numpy array, size [batch_size x 784]
        :return: probability distributions of the batch, tensor, size [batch_size x 10]
        '''
        output = self.ff_layer1(th.tensor(inputs, dtype=th.float32))
        output = th.squeeze(output)
        output = F.relu(output)
        output = self.ff_layer2(output)
        return output
    
    def loss_function(self, logits, labels):
        '''
        :param probs: probability distributions of the batch, tensor, size [batch_size x 10]
        :param labels: the correct classifications of the batch, numpy array, size [batch_size]
        :return: loss of the batch
        '''
        labels = th.tensor(labels, dtype=th.long)
        l = self.loss_layer(logits, labels)
        return l

def train(model, train_inputs, train_labels):
    '''
    :param model: The model we build
    :param train_inputs: training inputs, numpy array, size [60000, 784], dtype=float
    :param train_labels: training labels, numpy array, size [60000,], dtype=int
    :return: list of losses across batches
    '''
    l_list = []
    num_iter = train_inputs.shape[0] // model.batch_size
    for i in range(num_iter):
        model.optimizer.zero_grad() 
        s = np.random.shuffle(np.arange(model.batch_size))
        batch_inputs = train_inputs[i * model.batch_size: (i + 1) * model.batch_size, :]
        batch_labels = train_labels[i * model.batch_size: (i + 1) * model.batch_size]
        batch_inputs = th.autograd.Variable(th.tensor(batch_inputs[s], dtype=th.float))
        batch_labels = th.autograd.Variable(th.LongTensor(np.squeeze(batch_labels[s])))
        batch_logits = model.call(batch_inputs)
        batch_loss = model.loss_function(batch_logits, batch_labels)
        l_list.append(batch_loss.item())
        batch_loss.backward() 
        model.optimizer.step()
    return l_list

def train2(model, train_dataset, device):
    '''
    :param model: The model we build
    :param train_dataset: the dataset loaded in if not previously downloaded
    :param device: 'cpu' or 'cuda' (if available)
    :return: list of losses across batches
    '''
    l_list = []
    for batch_inputs, batch_labels in train_dataset:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        model.optimizer.zero_grad()
        batch_inputs = batch_inputs.view(-1, 784)
        batch_logits = model.call(batch_inputs)
        batch_loss = model.loss_function(batch_logits, batch_labels)
        l_list.append(batch_loss.item())
        batch_loss.backward() 
        model.optimizer.step()
    return l_list

def test(model, test_inputs, test_labels):
    logits = model.call(th.tensor(test_inputs, dtype=th.float))
    pred = th.argmax(logits, dim=1).numpy()
    correct = np.sum(np.equal(pred, test_labels))
    return correct / test_inputs.shape[0]

def test2(model, test_dataset, device):
    correct = 0
    counter = 0
    for batch_inputs, batch_labels in test_dataset:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        batch_inputs = batch_inputs.view(-1, 784)
        counter += list(batch_inputs.size())[0]
        logits = model.call(batch_inputs)
        pred = th.argmax(logits, dim=1).numpy()
        correct += np.sum(np.equal(pred, batch_labels.numpy()))
    return correct / counter

def visualize_loss(l_list):
    plt.plot(range(len(l_list)), l_list)
    plt.ylabel('loss')
    plt.xlabel('batch number')
    plt.show()

def main():
    answer = input("Enter Mode: local_mode (type 1), use pytorch_dataset (type 2): ")
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    flag = True
    model = Model()
    if answer == "1":
        train_inputs, train_labels = preprocess("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz", 60000)
        test_inputs, test_labels = preprocess("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz", 10000)
    else:
        flag = False
        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor())
        train_dataset = data.DataLoader(dataset=train_dataset, batch_size=model.batch_size, shuffle=True)
        test_dataset = data.DataLoader(dataset=test_dataset, batch_size=model.batch_size, shuffle=False)
    
    l_list = []
    for epoch in range(10):
        print(f"Epoch {epoch+1} now! ")
        if flag:
            l_list += train(model, train_inputs, train_labels)
        else:
            l_list += train2(model, train_dataset, device)
    visualize_loss(l_list)
    if flag:
        acc = test(model, test_inputs, test_labels)
    else:
        acc = test2(model, test_dataset, device)
    print(acc)
    

if __name__ == "__main__":
    main()