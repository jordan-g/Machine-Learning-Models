'''
PyTorch implementation of an RNN with two LSTM hidden layers trained to
predict the language of orgin of surnames. Adapted from:

https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
'''

import unicodedata
import string
import random
import glob
import io
import os
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_letters = string.ascii_letters + " .,;'"
n_letters   = len(all_letters)

# convert a Unicode string to ASCII
def unicode_to_ascii(string):
    return "".join( c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn' and c in all_letters)

# read a file and split it into lines
def read_lines(filename):
    lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [ unicode_to_ascii(line) for line in lines ]

# construct a string with the amount of elapsed time since a given time
def time_since(previous_time):
    current_time = time.time()
    s = current_time - previous_time
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

for filename in glob.glob('data/surnames/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

# separate 10% of training examples into a testing set
test_category_lines = {}
for category in category_lines.keys():
    lines = category_lines[category]

    n_test_lines = int(0.1*len(lines))

    test_lines = lines[:n_test_lines]

    test_category_lines[category] = test_lines

    del lines[:n_test_lines]

n_categories = len(all_categories)

# find the letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)

# turn a line into a <line_length x 1 x n_letters> tensor
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

# convert an output vector into a category, which is determined by
# the index of the element of the output vector with maximum value
def output_to_category(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# randomly choose an element of the list l
def random_choice(l):
    return l[random.randint(0, len(l) - 1)]

# randomly choose a training example
def random_training_example():
    category        = random_choice(all_categories)
    line            = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor     = line_to_tensor(line)

    return category, line, category_tensor, line_tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(RNN, self).__init__()
        # store input and hidden layer size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.input_size  = input_size

        # create LSTM hidden layers
        self.lstm_1 = torch.nn.LSTM(input_size, hidden_size_1)
        self.lstm_2 = torch.nn.LSTM(hidden_size_1, hidden_size_2)

        # create output layer
        self.output  = nn.Linear(hidden_size_2, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

        # initialize hidden states
        self.reset_hidden_states()

    def forward(self, input):
        # get hidden layer output and update its hidden state
        hidden_1, self.hidden_state_1 = self.lstm_1(input.view(1, 1, self.input_size), self.hidden_state_1)
        hidden_2, self.hidden_state_2 = self.lstm_2(hidden_1, self.hidden_state_2)

        # compute output
        output = self.output(hidden_2)
        output = self.dropout(output)
        output = self.softmax(output.view(1, output.size()[-1]))

        return output

    def reset_hidden_states(self):
        # reset LSTM layers' hidden states
        self.hidden_state_1 = (torch.randn(1, 1, self.hidden_size_1), torch.randn(1, 1, self.hidden_size_1))
        self.hidden_state_2 = (torch.randn(1, 1, self.hidden_size_2), torch.randn(1, 1, self.hidden_size_2))

# set hidden layer size
hidden_size_1 = 128
hidden_size_2 = 64

# create RNN
rnn = RNN(n_letters, hidden_size_1, hidden_size_2, n_categories)

# set loss function
criterion = nn.NLLLoss()

# set learning rate
learning_rate = 0.1

# define training function for one training example
def train(category_tensor, line_tensor):
    # reset LSTM layers' hidden states
    rnn.reset_hidden_states()

    # zero out the gradient
    rnn.zero_grad()

    # run the RNN through the sequence of characters in the training example
    for i in range(line_tensor.size()[0]):
        output = rnn(line_tensor[i])

    # do gradient descent
    loss = criterion(output, category_tensor)
    loss.backward()

    # add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

# return the output of the RNN given an input tensor
def evaluate(input_tensor):
    # reset LSTM layers' hidden states
    rnn.reset_hidden_states()

    # run the RNN through the sequence of characters in the training example
    for i in range(input_tensor.size()[0]):
        output = rnn(input_tensor[i])

    return output

# compute the error on the test set
def test():
    n_correct = 0
    n_test_examples = 0
    for category in test_category_lines.keys():
        lines = test_category_lines[category]

        for line in lines:
            with torch.no_grad():
                output = evaluate(line_to_tensor(line))

                n_test_examples += 1

                guess, guess_i = output_to_category(output)
                if guess == category:
                    n_correct += 1

    test_error = 100*(1 - n_correct/n_test_examples)

    return test_error

# plot a confusion matrix for the test data
def plot_confusion_matrix():
    # keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)

    for category in test_category_lines.keys():
        lines = test_category_lines[category]

        for line in lines:
            with torch.no_grad():
                output                          = evaluate(line_to_tensor(line))
                guess, guess_i                  = output_to_category(output)
                category_i                      = all_categories.index(category)
                confusion[category_i][guess_i] += 1

    # normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # force a label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.show()

# print the top n_predictions categories predicted by the RNN given an input line
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(line_to_tensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

# set number of training examples to show
n_iters = 200000

# set printing and plotting parameters
print_every = 5000
plot_every  = 1000

# keep track of losses for plotting
current_loss = 0
all_losses = []

# plot initial confusion matrix
plot_confusion_matrix()

# record start time
start_time = time.time()

for iter in range(1, n_iters + 1):
    # get a random training example
    category, line, category_tensor, line_tensor = random_training_example()

    # train on this training example
    output, loss = train(category_tensor, line_tensor)

    # update average loss over plot_every training examples
    current_loss += loss

    # add the average loss over the last plot_every training examples to the list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    # print a progress update every print_every training examples
    if iter % print_every == 0:
        test_error = test()

        guess, guess_i = output_to_category(output)
        correct = '✓' if guess == category else '✗ (%s)' % category

        print("Ex. {0:>10} ({1:>3}%). Time elapsed: {2:>7}. Loss: {3:.4f}. Test error: {4:.2f}%. {5} / {6} {7}".format(iter, int(iter/n_iters*100), time_since(start_time), all_losses[-1], test_error, line, guess, correct))

# plot loss
plt.figure()
plt.plot(all_losses)
plt.show()

# plot final confusion matrix
plot_confusion_matrix()

predict('Dostoevsky')
predict('Jackson')
predict('Motohashi')
predict("Miyazaki")
predict("Villeneuve")