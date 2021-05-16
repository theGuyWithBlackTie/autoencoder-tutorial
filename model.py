import torch.nn as nn
import torch
import torch.nn.functional as F

class autoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(autoEncoder, self).__init__()

        self.input_layer              = nn.Linear(input_dim, 75)

        self.encoder_hidden_layer_one = nn.Linear(75, 25)
        self.encoder_hidden_layer_two = nn.Linear(25,2)

        self.decoder_hidden_layer_one = nn.Linear(2, 25)
        self.decoder_hidden_layer_two = nn.Linear(25, 75)

        self.output_layer             = nn.Linear(75,100)


    def forward(self, input):
        output = (self.input_layer(input))

        output = self.encoder_hidden_layer_one(output)
        embed  = self.encoder_hidden_layer_two(output)

        output = self.decoder_hidden_layer_one(embed)
        output = self.decoder_hidden_layer_two(output)

        output = self.output_layer(output)

        return output, embed
