import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

matplotlib_is_available = True
try:
  from matplotlib import pyplot as plt
except ImportError:
  print("Will skip plotting; matplotlib is not available.")
  matplotlib_is_available = False

# data parameters
data_mean   = 4
data_stddev = 1.25

def data_sample(mu, sigma, size):
    return sigma*torch.randn(size) + mu

def input_sample(m, n):
    return torch.rand(m, n)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        nn.Module.__init__(self)

        self.map_1 = nn.Linear(input_size, hidden_size)
        self.map_2 = nn.Linear(hidden_size, hidden_size)
        self.map_3 = nn.Linear(hidden_size, output_size)

        self.f = f

    def forward(self, x):
        x = self.map_1(x)
        x = self.f(x)
        x = self.map_2(x)
        x = self.f(x)
        x = self.map_3(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        nn.Module.__init__(self)

        self.map_1 = nn.Linear(input_size, hidden_size)
        self.map_2 = nn.Linear(hidden_size, hidden_size)
        self.map_3 = nn.Linear(hidden_size, output_size)

        self.f = f

    def forward(self, x):
        x = self.map_1(x)
        x = self.f(x)
        x = self.map_2(x)
        x = self.f(x)
        x = self.map_3(x)
        x = self.f(x)

        return x

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def train():
    # model parameters
    g_input_size   = 1     # Random noise dimension coming into generator, per output vector
    g_hidden_size  = 5     # Generator complexity
    g_output_size  = 1     # Size of generated output vector
    d_input_size   = 500   # Minibatch size - cardinality of distributions
    d_hidden_size  = 10    # Discriminator complexity
    d_output_size  = 1     # Single dimension for 'real' vs. 'fake' classification
    minibatch_size = d_input_size

    d_learning_rate = 1e-3
    g_learning_rate = 1e-3
    sgd_momentum = 0.9

    num_epochs = 5000
    print_interval = 100
    d_steps = 20
    g_steps = 20

    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.tanh

    G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size,
                  f=generator_activation_function)
    D = Discriminator(input_size=d_input_size,
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=discriminator_activation_function)

    criterion = nn.BCELoss()

    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)

    for epoch in range(num_epochs):
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            d_real_data = Variable(data_sample(data_mean, data_stddev, d_input_size))
            d_real_decision = D(d_real_data)
            d_real_error = criterion(d_real_decision, Variable(torch.ones([1,1])))  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_gen_input = Variable(input_sample(minibatch_size, g_input_size))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data.t())
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,1])))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(input_sample(minibatch_size, g_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data.t())
            g_error = criterion(dg_fake_decision, Variable(torch.ones([1,1])))  # Train G to pretend it's genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
            ge = extract(g_error)[0]

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
                  (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))

    if matplotlib_is_available:
        print("Plotting the generated distribution...")
        values = extract(g_fake_data)
        print(" Values: %s" % (str(values)))
        plt.hist(values, bins=50)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram of Generated Distribution')
        plt.grid(True)
        plt.show()

train()