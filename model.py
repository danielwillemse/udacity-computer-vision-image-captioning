import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.4):
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, dropout=drop_prob)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1] 
        
        embeds = self.word_embeddings(captions)

        inputs = torch.cat((features.unsqueeze(1), embeds), dim=1)

        self.hidden = self.init_hidden(features.shape[0])
        
        outputs, self.hidden = self.lstm(inputs, self.hidden)

        outputs = self.fc(outputs)
        
        return outputs

    def init_hidden(self, batch_size):
        return (
            torch.normal(0, 0.5, size=(self.num_layers, batch_size, self.hidden_size)).cuda(),
            torch.normal(0, 0.5, size=(self.num_layers, batch_size, self.hidden_size)).cuda()
        )
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        sentence = []
        
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            outputs = self.fc(outputs)

            _, output_index = torch.max(outputs, 2)
            token_index = output_index.item()
            sentence.append(token_index)
            
            inputs = self.word_embeddings(output_index)
            
            if (token_index == 1): # <end> token
                break

            
        return sentence