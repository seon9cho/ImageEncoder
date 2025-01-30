import torch
import torch.nn as nn
import math


class ImageEncoder(nn.Module):
    def __init__(self, d_model, num_layers, f, d, activation=nn.ReLU, dropout=0.1):
        super(ImageEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d = d
        self.ff = nn.Linear(d_model, f*d_model)
        F = f*d_model
        self.conv_in = nn.Conv2d(\
            d_model*f // (d**2), F, 3, stride=1, padding=0
        )
        self.bn = nn.ModuleList(\
            nn.BatchNorm2d(F//(2**i)) for i in range(1, num_layers+1)
        )
        self.upsample = nn.ModuleList(\
            nn.Upsample(scale_factor=2, 
            mode="bilinear", 
            align_corners=False) 
            for i in range(num_layers)
        )
        self.convs = nn.ModuleList(\
            nn.Conv2d(F//(2**i), 
            F//(2**(i+1)), 
            3, 
            stride=1, 
            padding=0) 
            for i in range(num_layers)
        )
        
        self.conv_out = nn.Conv2d(\
            F//(2**num_layers), 3, 3, stride=1, padding=0
        )

        self.pad = nn.ReflectionPad2d(1)
        self.activation = activation()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.ff(x)
        x = x.reshape(batch_size, -1, self.d, self.d)
        x = self.pad(x)
        x = self.conv_in(x)
        for i in range(self.num_layers):
            x = self.upsample[i](x)
            x = self.pad(x)
            x = self.convs[i](x)
            x = self.dropout(x)
            x = self.activation(x)
            x = self.bn[i](x)
        x = self.pad(x)
        x = self.tanh(self.conv_out(x))
        return x

class ImageDecoder(nn.Module):
    def __init__(self, d_model, num_layers, activation=nn.ReLU, dropout=0.1):
        super(ImageDecoder, self).__init__()
        self.num_layers = num_layers
        init_dim = d_model // (2**(num_layers))
        self.d_model = d_model
        self.initial_conv = nn.Conv2d(3, init_dim, 3, stride=1, padding=1)
        self.down_sample = nn.ModuleList(\
            nn.Conv2d(init_dim*(2**(i)), 
            init_dim*(2**(i+1)), 
            2, 
            stride=2) 
            for i in range(num_layers)
        )
        self.conv = nn.ModuleList(\
            nn.Conv2d(init_dim*(2**(i+1)), 
            init_dim*(2**(i+1)), 
            3, 
            stride=1, 
            padding=1) 
            for i in range(num_layers)
        )
        self.bn = nn.ModuleList(\
            nn.BatchNorm2d(init_dim*(2**(i+1))) 
            for i in range(num_layers)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation()
        
    def forward(self, x):
        x = self.initial_conv(x)
        for i in range(self.num_layers):
            x = self.down_sample[i](x)
            x = self.dropout(x)
            x = self.activation(x)
            x = self.bn[i](x)
            x = self.conv[i](x)
            x = self.dropout(x)
            x = self.activation(x)
        return x.view(x.size(0), -1, self.d_model)

def tensor2image(t):
    image = t.permute(1, 2, 0).cpu().detach().numpy()
    return (image + 1) / 2