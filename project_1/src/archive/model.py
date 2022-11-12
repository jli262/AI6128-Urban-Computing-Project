import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from data import split_floor_data
from sklearn.preprocessing import StandardScaler

class Encoder(nn.Module):
    def __init__(self, wifi_dim, ibeacon_dim, output_dim=32, hidden_magn=32, hidden_wifi=64, hidden_ibeacon=64, drop_rate=0.4, actfunc=nn.ReLU, use_wifi=True, use_ibeacon=True):
        super(Encoder, self).__init__()
        self.activate_function = actfunc
        self.wifi_dim = wifi_dim 
        self.ibeacon_dim = ibeacon_dim
        self.feature_dim = 4+hidden_magn+(hidden_ibeacon+1 if use_ibeacon else 0)+(hidden_wifi+1 if use_wifi else 0)
        self.use_wifi = use_wifi
        self.use_ibeacon = use_ibeacon

        self.magn_encoder = nn.Sequential(
            nn.Linear(4, hidden_magn*2),
            nn.BatchNorm1d(hidden_magn*2),
            nn.Dropout(drop_rate*0.25),
            actfunc(),
            nn.Linear(hidden_magn*2, hidden_magn),
            )

        self.wifi_encoder = nn.Sequential(
            nn.Linear(wifi_dim+1, hidden_wifi*4),
            nn.BatchNorm1d(hidden_wifi*4),
            nn.Dropout(drop_rate*0.5),
            actfunc(),
            nn.Linear(hidden_wifi*4, hidden_wifi*2),
            nn.BatchNorm1d(hidden_wifi*2),
            nn.Dropout(drop_rate),
            actfunc(),
            nn.Linear(hidden_wifi*2, hidden_wifi),
            )
        
        self.ibeacon_encoder = nn.Sequential(
            nn.Linear(ibeacon_dim+1, hidden_ibeacon*4),
            nn.BatchNorm1d(hidden_ibeacon*4),
            nn.Dropout(drop_rate*0.5),
            actfunc(),
            nn.Linear(hidden_ibeacon*4, hidden_ibeacon*2),
            nn.BatchNorm1d(hidden_ibeacon*2),
            nn.Dropout(drop_rate),
            actfunc(),
            nn.Linear(hidden_ibeacon*2, hidden_ibeacon),
            )

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            actfunc(),
            nn.Linear(self.feature_dim, output_dim*4),
            nn.BatchNorm1d(output_dim*4),
            nn.Dropout(drop_rate*0.5),
            actfunc(),
            nn.Linear(output_dim*4, output_dim*2),
            nn.BatchNorm1d(output_dim*2),
            actfunc(), 
            nn.Linear(output_dim*2, output_dim),
            )

    def forward(self, x):
        magn_o, wifi_det, ibeacon_det, wifi_o, ibeacon_o = x.split([4,1,1,self.wifi_dim, self.ibeacon_dim], dim=1)
        
        magn_out = self.magn_encoder(magn_o)

        if self.use_wifi:
            wifi = torch.cat([wifi_det, wifi_o], dim=1)
            wifi_out = self.wifi_encoder(wifi)
        if self.use_ibeacon:
            ibeacon = torch.cat([ibeacon_det, ibeacon_o], dim=1)
            ibeacon_out = self.ibeacon_encoder(ibeacon)

        if self.use_wifi:
            if self.use_ibeacon:
                output = torch.cat([magn_o, magn_out, wifi_out, ibeacon_out, wifi_det, ibeacon_det], dim=1)
            else:
                output = torch.cat([magn_o, magn_out, wifi_out, wifi_det], dim=1)
        else:
            if self.use_ibeacon:
                output = torch.cat([magn_o, magn_out, ibeacon_out, ibeacon_det], dim=1)
            else:
                output = torch.cat([magn_o, magn_out], dim=1)

        output = self.encoder(output)

        return output
