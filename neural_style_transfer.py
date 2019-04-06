'''
PyTorch implementation of neural style transfer. Adapted from:

https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image (use a small image size if a GPU is not available)
image_size = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize(image_size), # scale imported image
    transforms.ToTensor()])        # transform it into a torch tensor

# load an image from a given filename to a tensor
def load_image_to_tensor(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# load style and content images
style_image   = load_image_to_tensor("./data/images/neural-style/picasso.jpg")
content_image = load_image_to_tensor("./data/images/neural-style/dancing.jpg")

# make sure they're the same size
assert style_image.size() == content_image.size(), "Style and content images must be the same size."

# plot a tensor representing an image
def imshow(tensor, title=None):
    plt.figure()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

# plot style image
imshow(style_image, title='Style Image')

# plot content image
imshow(content_image, title='Content Image')

# load input image
input_image = load_image_to_tensor("./data/images/neural-style/dancing.jpg")

# plot input image
imshow(input_image, title='Input Image')

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # detach the target from the autograd tree
        self.target = target.detach()

    def forward(self, input):
        # update the content loss
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    # a = batch size (=1)
    # b = number of feature maps
    # (c, d) = dimensions of a feature map
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)

    # compute the Gram matrix
    G = torch.mm(features, features.t())

    # normalize the values of the Gram matrix by dividing by the number of element in each feature map
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        # compute the gram matrix
        G = gram_matrix(input)

        # update the style loss
        self.loss = F.mse_loss(G, self.target)
        return input

# load a pre-trained 19-layer VGG CNN
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# set normalization means and standard deviations for each channel
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize an input image
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, image):
        # normalize the image
        return (image - self.mean) / self.std

# set desired depth layers to compute style/content losses
content_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_layers_default   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_image, content_image, content_layers=content_layers_default, style_layers=style_layers_default):
    # copy the CNN
    cnn = copy.deepcopy(cnn)

    # initialize the normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # initialize lists of content & style losses
    content_losses = []
    style_losses   = []

    # create a new nn.Sequential to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a convolutional layer
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        # add the layer to our model
        model.add_module(name, layer)

        if name in content_layers:
            # create content loss
            target = model(content_image).detach()
            content_loss = ContentLoss(target)

            # add content loss module
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # create style loss
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)

            # add style loss module
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # remove layers afer the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_image):
    # create an optimizer for training the input image
    optimizer = optim.LBFGS([input_image.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std, content_image, style_image, input_image, n_iterations=1000, style_weight=1000000, content_weight=1):
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_image, content_image)
    optimizer = get_input_optimizer(input_image)

    print('Optimizing...')
    iteration = [0]
    while iteration[0] <= n_iterations:
        def closure():
            # clip the values of the updated input image
            input_image.data.clamp_(0, 1)

            # run the input image through the model
            optimizer.zero_grad()
            model(input_image)

            # compute scores for style and content
            style_score   = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score   *= style_weight
            content_score *= content_weight

            # compute final loss and do gradient descent on the input image
            loss = style_score + content_score
            loss.backward()

            iteration[0] += 1
            if iteration[0] % 50 == 0:
                print("Iteration {0:>4}. Style loss: {1:.4f}. Content loss: {2:.4f}.".format(iteration[0], style_score.item(), content_score.item()))

            return loss

        optimizer.step(closure)

    # clip the values of the final input image
    input_image.data.clamp_(0, 1)

    return input_image

# run style transfer
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_image, style_image, input_image)

# show the final image
imshow(output, title='Output Image')