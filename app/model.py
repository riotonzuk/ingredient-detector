import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, n_class):
        super(ConvNet,self).__init__()

        self.encoder = nn.Sequential(
            # layer 1 -- 224 x 224
            nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # layer 2 -- 112 x 112
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # layer 3 -- 56 x 56
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # layer 4 -- 28 x 28
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # layer 5 -- 14 x 14
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Flatten(),
            nn.LazyLinear(n_class)
        )

        self.linear = nn.LazyLinear(1024)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),


            # layer 2 -- 112 x 112
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),


            # layer 3 -- 56 x 56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),


            # layer 4 -- 28 x 28
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),


            # layer 5 -- hopefully 3 x 224 x 224
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(size=(224, 224))
        )

    def forward(self, x):
        ingredient_pred = self.encoder(x)
        x = self.linear(ingredient_pred)
        # x is N x 1024 -> (1, 32, 32)
        # N is number of samples
        x = x.reshape(len(x), 1, 32, 32)
        image_pred = self.decoder(x)
        return image_pred, ingredient_pred


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_conditioned_generator = \
        nn.Sequential(nn.LazyLinear(1024),
                      nn.Linear(1024, 16))

        self.latent = \
        nn.Sequential(nn.LazyLinear(4*4*512),
                      nn.LeakyReLU(0.2, inplace=True))


        self.model = \
        nn.Sequential(nn.ConvTranspose2d(513, 64*8, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(64*8, momentum=0.1,  eps=0.8),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1,bias=False),
                      nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1,bias=False),
                      nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64*2, 64*1, 4, 2, 1,bias=False),
                      nn.BatchNorm2d(64*1, momentum=0.1,  eps=0.8),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64*1, 3, 4, 2, 1, bias=False),
                      nn.Tanh())

    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512,4,4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        #print(image.size())
        return image