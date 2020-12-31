import torch as th
import torch.nn.functional as F
import torchtext
from torchtext.data import Field, BucketIterator
import spacy
import time
import numpy as np
from matplotlib import pyplot as plt

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

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
    return len(source.vocab), len(target.vocab), train_iterator, valid_iterator, test_iterator, trg_pad_index

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

def epoch_time(start, end):
    'Get the training and validating time for an epoch'
    t = end - start
    minutes = int(t // 60)
    seconds = int(t - minutes * 60)
    return minutes, seconds

def visualize_loss(l_list):
    '''
    Visualize the losses across batches
    :param l_list: List of losses across batches
    '''
    plt.plot(range(len(l_list)), l_list)
    plt.ylabel('Loss')
    plt.xlabel('Batch number')
    plt.show()

def train(model, train_data, criterion):
    loss_list = []
    for i, batch in enumerate(train_data):
        model.optimizer.zero_grad()
        output = model(batch.src, batch.trg)
        batch_loss = criterion(output, th.transpose(batch.trg, dim0=0, dim1=1))
        loss_list.append(batch_loss.item())
        batch_loss.backward()
        model.optimizer.step()
    return loss_list

def evaluate(model, data, criterion):
    loss_list = []
    with th.no_grad():
        for i, batch in enumerate(data):
            output = model(batch.src, batch.trg)
            batch_loss = criterion(output, th.transpose(batch.trg, dim0=0, dim1=1))
            loss_list.append(batch_loss.item())
    return sum(loss_list) / len(loss_list)

class seq2seq(th.nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, rnn_size, embedding_size, num_layers, dropout):
        super(seq2seq, self).__init__()

        # Hyperparameter Initialization
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.p_dropout = dropout

        # Trainable parameters
        self.src_embedding_layer = th.nn.Embedding(src_vocab_size, embedding_dim=self.embedding_size)
        self.trg_embedding_layer = th.nn.Embedding(trg_vocab_size, embedding_dim=self.embedding_size)
        self.encoder = th.nn.LSTM(input_size=self.embedding_size, hidden_size=self.rnn_size, num_layers=self.num_layers)
        self.decoder = th.nn.LSTM(input_size=self.embedding_size, hidden_size=self.rnn_size, num_layers=self.num_layers)
        self.dropout = th.nn.Dropout(self.p_dropout)
        self.ff_layer = th.nn.Linear(in_features=self.rnn_size, out_features=self.trg_vocab_size)
        self.softmax_layer = th.nn.LogSoftmax(dim=2)
        self.optimizer = th.optim.Adam(self.parameters())
    
    def forward(self, encoder_input, decoder_input):
        '''
        :param encoder_input: batched indices in src language
        :param decoder_input: batched indices in trg language
        :return probs: The 3d probabilities as a tensor, [trg_window_size x batch_size x trg_vocab_size]
        '''
        src_embeddings = self.src_embedding_layer(encoder_input)
        trg_embeddings = self.trg_embedding_layer(decoder_input)
        _, (encoded_memory_state, encoded_carrier_state) = self.encoder(self.dropout(src_embeddings))
        decoded, (_, _) = self.decoder(self.dropout(trg_embeddings), (encoded_memory_state, encoded_carrier_state))
        probs = self.softmax_layer(self.ff_layer(decoded))
        probs = th.transpose(probs, dim0=0, dim1=1)
        probs = th.transpose(probs, dim0=1, dim1=2)
        return probs

def main():
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    batch_size = 50
    num_layers = 2
    rnn_size = 200
    embedding_size = 256
    p_dropout = 0.5
    src_vocab_size, trg_vocab_size, train_data, valid_data, test_data, trg_pad_index = preprocess(device, batch_size)
    seq2seqmodel = seq2seq(src_vocab_size, trg_vocab_size, rnn_size, embedding_size, num_layers, p_dropout)
    criterion = th.nn.NLLLoss(ignore_index=trg_pad_index)
    loss_list = []
    valid_list = []
    for epoch in range(2):
        start = time.time()
        l1 = train(seq2seqmodel, train_data, criterion)
        loss_list += l1
        l2 = evaluate(seq2seqmodel, valid_data, criterion)
        valid_list.append(l2)
        end = time.time()
        minutes, seconds = epoch_time(start, end)
        print(f'Epoch {epoch+1}, time used: {minutes}m{seconds}sec.')
        print(f'Train loss {sum(l1)/len(l1):.3f}, train perplexity {np.exp(sum(l1)/len(l1)):.3f}.')
        print(f'Valid loss {l2:.3f}, valid perplexity {np.exp(l2):.3f}.')
        if l2 <= min(valid_list):
            th.save(seq2seqmodel.state_dict(), 'seq2.pt')
    visualize_loss(loss_list)
    seq2seqmodel.load_state_dict(th.load('seq2.pt'))
    test_loss = evaluate(seq2seqmodel, test_data, criterion)
    print(f'Test loss {test_loss:.3f}, test perplexity {np.exp(test_loss):.3f}')

if __name__ == "__main__":
    main()