
from ops import lrelu

_EPS = 1e-5
import torch.nn as nn
import torch 

class ContentPredictorBuilder(nn.Module):
    def __init__(self, config):
        super(ContentPredictorBuilder, self).__init__()
        
        self.config=config
        self.z_dim = config["model_params"]["z_dim"]
        self.aux_dim = config["model_params"]["aux_dim"]
        
        self.fc1 = nn.Linear(512, self.aux_dim)
        self.bn1 = nn.BatchNorm1d(1, eps=1e-05, momentum=0.9)
        self.fc2 = nn.Linear(self.aux_dim, self.z_dim)
            
    def forward(self, h, is_training):
        h = self.fc1(h)
        h = self.bn1(h)
        h = lrelu(h)
        h = self.fc2(h)
        return torch.tanh(h)
