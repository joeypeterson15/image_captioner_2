import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1] # remove classification module(softmax)
        self.resnet = nn.Sequential(*modules)
        # resnet.fc = resnets fully connected layer (last layer of resnet)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1) # flatten feature maps
        features = self.embed(features) # embedded vector
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        '''
        Args:
            embed_size: final embedding size of the CNN encoder
            hidden_size: hidden size of the LSTM
            vocab_size: size of the vocabulary
            num_layers: number of layers of the LSTM
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))


    def forward(self, features, captions):
        """
        Args:
            features: features tensor, shape is (batch size, embed_size(output of feature embedding tensor from encoder))
            captions: captions tensor, shape is (batch size, cap_length(maximum length of the caption))
        Returns:
            outputs: scores of the linear layer
        """

        # remove <end> token from captions and embed captions
        cap_embedding = self.embed(
            captions[:, :-1]
        )

        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)

        lstm_out, self.hidden = self.lstm(
            embeddings
        )

        output = self.linear(lstm_out)

        return output
    
    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor(inputs) and returns predicted
        sentence (list of tensor ids of length max_len)
        Args:
            inputs: shape is (1, 1, embed_size)
            states: initial hidden state of the LSTM
            max_len: maximum length of the predicted sentence

        Returns:
            res: list of predicted word indices
        """

        res = []

        # Now we feed the LSTM output and hidden states back into 
        for i in range(max_len):
            lstm_out, states = self.lstm(
                inputs, states
            )
            outputs = self.linear(lstm_out.squeeze(dim=1))
            _, predicted_idx = outputs.max(dim=1)
            res.append(predicted_idx.item())

            if predicted_idx == 1: # break if stop index
                break
            inputs = self.embed(predicted_idx)
            # prepare input for next iteration
            inputs = inputs.unsqueeze(1) # inputs: (1, 1, embed_size)