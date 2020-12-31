import torch as th
import torch.nn.functional as F
import torchtext
from torchtext.data import Field, BucketIterator
import numpy as np
from matplotlib import pyplot as plt
import spacy
import time

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

class seq2seq(th.nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, batch_size, trg_pad_index):
        super(seq2seq, self).__init__()

        # Hyperparameter Initialization
        self.batch_size = batch_size
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.trg_pad_index = trg_pad_index
        self.learning_rate = 0.001
        self.embedding_size = 256
        self.rnn_size = 200

        # Trainable Parameters Initialization
        self.src_embedding = th.nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=self.embedding_size)
        self.trg_embedding = th.nn.Embedding(num_embeddings=trg_vocab_size, embedding_dim=self.embedding_size)
        self.encoder = th.nn.LSTM(input_size=self.embedding_size, hidden_size=self.rnn_size, num_layers=2)
        self.decoder = th.nn.LSTM(input_size=self.embedding_size, hidden_size=self.rnn_size, num_layers=2)
        self.ff_layer = th.nn.Linear(in_features=self.rnn_size, out_features=self.trg_vocab_size)
        self.softmax_layer = th.nn.LogSoftmax(dim=2)
        
        # Other
        self.loss_layer = th.nn.NLLLoss(ignore_index=self.trg_pad_index)
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def call(self, encoder_input, decoder_input):
        '''
        :param encoder_input: batched ids corresponding to sentences in source language, [src_window_size x batch_size]
        :param decoder_input: batched ids corresponding to sentences in target language, [trg_window_size x batch_size]
        :return probs: The 3d probabilities as a tensor, [trg_window_size x batch_size x trg_vocab_size]
        '''
        # Get Embeddings
        src_embeddings = self.src_embedding(encoder_input)
        trg_embeddings = self.trg_embedding(decoder_input)

        # Encode and Decode
        _, (encoder_memory_state, encoder_carrier_state) = self.encoder(src_embeddings)
        decoded, (_, _) = self.decoder(trg_embeddings, (encoder_memory_state, encoder_carrier_state))
        return self.softmax_layer(self.ff_layer(decoded))
    
    def predict(self, encoder_input):
        '''
        Attempt to make open-ended predictions. (failed)
        Open for discussion of implementation. Contact me. 
        '''
        # Get Embeddings
        src_embeddings = self.src_embedding(encoder_input)

        # Encode and Decode
        output, (encoder_memory_state, encoder_carrier_state) = self.encoder(src_embeddings)
        print(output.shape)
        decoded, (_, _) = self.decoder(output, (encoder_memory_state, encoder_carrier_state))
        return self.softmax_layer(self.ff_layer(decoded))

    def loss_function(self, probs, labels):
        '''
        :param probs: 3d probabilities as a tensor, [trg_window_size x batch_size x trg_vocab_size]
        :param labels: The expected translation in target language, [trg_window_size x batch_size]
        :return l: loss of the batch
        '''
        # probs should be transposed to [batch_size x trg_vocab_size x trg_window_size]
        probs = th.transpose(probs, dim0=0, dim1=1)
        probs = th.transpose(probs, dim0=1, dim1=2)
        labels = th.transpose(labels, dim0=0, dim1=1)
        return self.loss_layer(probs, labels)

    def accuracy(self, probs, labels):
        '''
        :param probs: 3d probabilities as a tensor, [trg_window_size x batch_size x trg_vocab_size]
        :param labels: The expected translation in target language, [trg_window_size x batch_size]
        :return acc: the accuracy of the batch
        '''
        preds = th.transpose(th.argmax(probs, dim=2), dim0=0, dim1=1).numpy()
        labels = th.transpose(labels, dim0=0, dim1=1).numpy()
        mask = np.where(labels == self.trg_pad_index, False, True)
        num = np.sum(mask)
        correct = np.sum(np.multiply(np.equal(preds, labels), mask))
        return correct / num

def preprocess(device, batch_size):
    '''
    Download and prepare data.
    :param device: 'cpu' or 'cuda' (if available)
    :param batch_size: batch size, set in main()
    :return: vocab sizes for source and target, 
            the train, valid, and test data, 
            and the index of the <pad> token in target vocab. 
    '''
    # Fetch Data
    source = Field(init_token='<sos>', eos_token='<eos>',tokenize=tokenize_de, lower=True)
    target = Field(init_token='<sos>', eos_token='<eos>',tokenize=tokenize_en, lower=True)
    train_data, valid_data, test_data = torchtext.datasets.Multi30k.splits(exts=('.de', '.en'), fields=(source, target), root='data/')
    # print(vars(train_data.examples[0]))

    # Build Vocab
    source.build_vocab(train_data, min_freq=2)
    target.build_vocab(train_data, min_freq=2)
    print('Vocab size of German (source) is ', len(source.vocab))
    print('Vocab size of English (target) is ', len(target.vocab))

    # Make Data Iterable
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size = batch_size, device = device)
    trg_pad_index = target.vocab.stoi[target.pad_token]
    return len(source.vocab), len(target.vocab), train_iterator, valid_iterator, test_iterator, trg_pad_index, source.vocab.stoi, target.vocab.stoi

def tokenize_de(text):
    '''
    The function for tokenize the German data and reverse it. 
    :param text: The text to be tokenized. 
    :return: List of tokenized strings.
    '''
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    '''
    The function for tokenize the English data and reverse it. 
    :param text: The text to be tokenized. 
    :return: List of tokenized strings.
    '''
    return [tok.text for tok in spacy_en.tokenizer(text)]

def train(model, train_data):
    '''
    Runs through one epoch all the training batches
    :param model: The model we build
    :param train_data: Data used for training
    :return l_list, num_non_pad: List of losses across batches, number of unpadded words
    '''
    l_list = []
    num_non_pad = 0
    for i, batch in enumerate(train_data):
        model.optimizer.zero_grad()
        probs = model.call(batch.src, batch.trg)
        batch_loss = model.loss_function(probs, batch.trg)
        num_non_pad += np.sum(np.where(batch.trg.numpy() == model.trg_pad_index, False, True))
        l_list.append(batch_loss.item())
        batch_loss.backward() 
        model.optimizer.step()
    return l_list, num_non_pad

def valid(model, valid_data):
    '''
    Get the accuracy of the model at the epoch. 
    '''
    for i, batch in enumerate(valid_data):
        with th.no_grad():
            probs = model.call(batch.src, batch.trg)
            return model.accuracy(probs, batch.trg)

def test(model, test_data, src_vocab, trg_vocab):
    '''
    Testing & visualize results
    '''
    for i, batch in enumerate(test_data):
        with th.no_grad():
            probs = model.call(batch.src, batch.trg)
            source = th.transpose(batch.src, dim0=0, dim1=1)
            labels = th.transpose(batch.trg, dim0=0, dim1=1)
            preds = th.transpose(th.argmax(probs, dim=2), dim0=0, dim1=1)
            s = [src_vocab[index][0] for index in source[0]]
            l = [trg_vocab[index][0] for index in labels[0]]
            r = [trg_vocab[index][0] for index in preds[0]]
            print(' '.join(s[::-1]))
            print(' '.join(l))
            print(' '.join(r))    

def visualize_loss(l_list):
    '''
    Visualize the losses across batches
    :param l_list: List of losses across batches
    '''
    plt.plot(range(len(l_list)), l_list)
    plt.ylabel('Loss')
    plt.xlabel('Batch number')
    plt.show()

def epoch_time(start, end):
    'Get the training and validating time for an epoch'
    t = end - start
    minutes = int(t // 60)
    seconds = int(t - minutes * 60)
    return minutes, seconds

def main():
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    batch_size = 50
    src_vocab_size, trg_vocab_size, train_data, valid_data, test_data, trg_pad_index, src_vocab, trg_vocab = preprocess(device, batch_size)
    model = seq2seq(src_vocab_size, trg_vocab_size, batch_size, trg_pad_index)
    # model.load_state_dict(th.load('seq2seqmodel.pt')) # use when running later
    l_list = []
    for epoch in range(2):
        start = time.time()
        l, num_non_pad= train(model, train_data)
        l_list += l
        p = np.exp(sum(l) / num_non_pad)
        acc = valid(model, valid_data)
        end = time.time()
        minutes, seconds = epoch_time(start, end)
        print(f'Epoch {epoch+1}, time used: {minutes}m{seconds}sec, perplexity is {p}, accuracy is {acc}.')
        th.save(model.state_dict(), 'Seq2SeqModel.pt')
    visualize_loss(l_list)
    l_src = list(src_vocab.items())
    l_trg = list(trg_vocab.items())
    test(model, test_data, l_src, l_trg)

if __name__ == "__main__":
    main()