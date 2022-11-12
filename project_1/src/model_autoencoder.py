import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os
import pandas as pd

from .preprocessing import Compute
from .data import FloorData

class Encoder(nn.Module):
    def __init__(self, wifi_dim, ibeacon_dim, output_dim=32, 
                    hidden_magn=32, hidden_wifi=64, hidden_ibeacon=64, 
                    drop_rate=0.4, actfunc=nn.ReLU, use_wifi=True, 
                    use_ibeacon=True):
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

class Decoder(nn.Module):
    def __init__(self, input_dim=32, hidden=64, drop_rate=0.2, actfunc=nn.Tanh):
        super(Decoder, self).__init__()
        self.activate_function = actfunc

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.Dropout(drop_rate),
            actfunc(),
            nn.Linear(hidden, hidden*2),
            nn.BatchNorm1d(hidden*2),
            nn.Dropout(drop_rate),
            actfunc(),
            nn.Linear(hidden*2, 2),
            )

    def forward(self, x):
        return self.decoder(x)


class DLnetwork(nn.Module):
    #def __init__(self, wifi_dim, ibeacon_dim, augmentation=True, use_wifi=True, use_ibeacon=True):
    def __init__(self, dimension_of_wifi_features: int, dimension_of_ibeacon_features: int, 
                    dimension_of_encoder_output: int, 
                    dimension_of_decoder_output: int,
                    dropout_rate: float, activation_function, 
                    magn_hidden_layer: int, use_wifi_features: bool, 
                    wifi_hidden_layer: int, use_ibeacon_features: bool, 
                    ibeacon_hidden_layer: int):
        super(DLnetwork, self).__init__()
        
        self.encoder = Encoder(dimension_of_wifi_features, dimension_of_ibeacon_features, 
                                    output_dim=dimension_of_encoder_output, 
                                    hidden_magn=magn_hidden_layer, 
                                    hidden_wifi=wifi_hidden_layer, 
                                    hidden_ibeacon=ibeacon_hidden_layer, 
                                    drop_rate=dropout_rate, actfunc=activation_function,
                                    use_wifi=use_wifi_features, use_ibeacon=use_ibeacon_features)
        self.decoder = Decoder(input_dim=dimension_of_encoder_output, 
                                    hidden=dimension_of_decoder_output, 
                                    drop_rate=dropout_rate, actfunc=activation_function)
    

    def forward(self, x):
        return self.decoder(self.encoder(x))


# class Encoder(nn.Module):
#     def __init__(self):
#         pass


# class Autoencoder(object):
#     def __init__(self):
#         pass


class FingerprintModel(object):
    def __init__(self, device: str = 'cuda', model_id: str = None):         
        self.model_id = model_id        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.epochs_current = 0
        self.loss_error = [[], [], []] # training loss, training error, test  error
        self.learning_rate = None
        self.data_train = None
        self.data_test = None

    def set_data(self, train: Dataset, test: Dataset, 
                    batch_size: int = 64) -> bool:
        y = train[:, :2]
        self.y_mean, self.y_std = torch.Tensor(y.mean(axis=0)).to(self.device), torch.Tensor(y.std(axis=0)).to(self.device)
        self.data_train = DataLoader(FloorData(train), batch_size = batch_size, shuffle = True)
        self.data_test = DataLoader(FloorData(test), batch_size = batch_size, shuffle = False)
        return True

    def initialize_model(self, dimension_of_wifi_features: int, 
                            dimension_of_ibeacon_features: int, 
                            for_augmented_data: bool = True,                             
                            use_wifi_features: bool = True, 
                            use_ibeacon_features: bool = True, 
                            ):
        dimension_of_encoder_output = 32
        dimension_of_decoder_output = 64
        activation_function = nn.ReLU
        dropout_rate = 0.0
        if for_augmented_data:       
            magn_hidden_layer = 32            
            wifi_hidden_layer = 128            
            ibeacon_hidden_layer = 128            
        else:
            magn_hidden_layer = 32            
            wifi_hidden_layer = 32         
            ibeacon_hidden_layer = 32   

        self.model = DLnetwork(dimension_of_wifi_features, 
                                dimension_of_ibeacon_features, 
                                dimension_of_encoder_output,
                                dimension_of_decoder_output,
                                dropout_rate,
                                activation_function,
                                magn_hidden_layer,
                                use_wifi_features,
                                wifi_hidden_layer,
                                use_ibeacon_features,
                                ibeacon_hidden_layer).to(self.device)

    def train(self, epochs, reduce_lr_epoch=5, startlr=0.1, verbose=True):
        if startlr is None:
            if self.lr is None:
                cur_lr = 0.01
            else:
                cur_lr = self.lr 
        else:
            cur_lr = startlr
            self.lr = cur_lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cur_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, 
                                                               patience=reduce_lr_epoch, cooldown=5, min_lr=1e-5)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(1, epochs+1):
            epoch_loss = 0
            epoch_error = 0
            batch_number = 0

            for x, y in iter(self.data_train):
                batch_number += 1 
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                pred = self.model(x)
                error = torch.sum(torch.sqrt(torch.sum((y-(pred*self.y_std + self.y_mean))**2, dim=1))) / y.shape[0]
                epoch_error += error.detach().item()

                loss = criterion(pred, (y-self.y_mean)/self.y_std)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

            mean_loss = epoch_loss / batch_number
            mean_error = epoch_error / batch_number
            test_error = self.evaluate()
            self.epochs_current += 1
            self.loss_error[0].append(mean_loss)
            self.loss_error[1].append(mean_error)
            self.loss_error[2].append(test_error.detach().item())
            if verbose:
              print(f'epoch: {self.epochs_current} | lr: {optimizer.param_groups[0]["lr"]} |epoch loss: {mean_loss} | \
                  Euclidean distance error on epoch: {mean_error}| Euclidean distance error on TestSet: {test_error} ')   
            #scheduler.step(mean_loss)
            scheduler.step(test_error.detach())
            
            if epoch > 100 and all(past_error <= test_error.detach().item() for past_error in list(self.loss_error[2])[-20:]):
              print(f'Early stopping at Epoch {epoch}')
              break

        return np.min(self.loss_error[2]), np.argmin(self.loss_error[2])+1, self.loss_error



    def evaluate(self):
        self.model.eval()
        total_error = 0
        batch_number = 0
        with torch.no_grad():
            for x, y in iter(self.data_test):
                batch_number += 1 
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                pred = self.model(x)
                error = torch.sum(torch.sqrt(torch.sum((y-(pred*self.y_std + self.y_mean))**2, dim=1))) / y.shape[0]
                total_error += error
        mean_error = total_error / batch_number
        return mean_error


    def plotloss(self):
        epochs = list(range(1, len(self.loss_error[0])+1))
        if not epochs:
            print(f'0 epoch trained')
            return

        plt.clf()
        fig = plt.figure(figsize=(12,8), dpi=80)
        loss_fig = fig.add_subplot(2,1,1)
        error_fig = fig.add_subplot(2,1,2)
        loss_fig.plot(epochs, self.loss_error[0], label='train loss',linestyle='-', color='red')
        error_fig.plot(epochs, self.loss_error[1], label='train error', linestyle='-', color='blue')
        error_fig.plot(epochs, self.loss_error[2], label='test error', linestyle='-', color='green')
        loss_fig.legend(loc='upper right')
        error_fig.legend(loc='upper right')
        plt.savefig('loss.jpg', bbox_inches='tight')
        plt.show()


    def predict(self, data, groundtruth=None):
        # data = {Mx, My, Mz, {bssid:rssi}, {uuid:rssi}}
        self.model.eval()
        x = torch.zeros([1,6+len(self.bssid2index)+len(self.uuid2index)])
        Mx, My, Mz = data[:3]
        MI = (Mx**2 + My**2 + Mz**2)**0.5
        Mx, My, Mz, MI = self.scaler.transform(np.array([[Mx,My,Mz,MI]]))[0]
        wifis = data[3]
        ibeacons = data[4]
        wifi_det, ibeacon_det = 1 if wifis else 0, 1 if ibeacons else 0
        x[0][:6] = torch.Tensor([[Mx, My, Mz, MI, wifi_det, ibeacon_det]])
        for bssid, rssi in wifis.items():
            if bssid in self.bssid2index:
                x[0][6+self.bssid2index[bssid]] = (100+rssi) / 100 
        for uuid, rssi in ibeacons.items():
            if uuid in self.uuid2index:
                x[0][6+len(self.bssid2index)+self.uuid2index[uuid]] = (100+rssi) / 100
        x = x.to(self.device)

        pred = self.model(x)
        pred = pred * self.y_std + self.y_mean
        pred = (float(pred[0][0]), float(pred[0][1]))

        plt.clf()
        plt.figure(figsize=(6,6), dpi=160)
        json_path = os.path.join('../data', self.site, self.floor, 'floor_info.json')
        with open(json_path) as file:
            mapinfo = json.load(file)['map_info']
        mapheight, mapwidth = mapinfo['height'], mapinfo['width']
        img = mpimg.imread(os.path.join('../data', self.site, self.floor, 'floor_image.png'))
        plt.imshow(img)
        mapscaler = (img.shape[0]/mapheight + img.shape[1]/mapwidth)/2
        pre = plt.scatter([pred[0]*mapscaler], [img.shape[0] - pred[1]*mapscaler], color='red', marker='o', s=7)
        if groundtruth:
            real = plt.scatter([groundtruth[0]*mapscaler], [img.shape[0] - groundtruth[1]*mapscaler], color='green', marker='x', s=7)
            plt.legend([pre, real], ['prediction', 'groundtruth'], loc='lower left')
        else:
            plt.legend([pre], ['prediction'], loc='lower left')
        plt.xticks((np.arange(25, mapwidth, 25) * mapscaler).astype('uint'), np.arange(25, mapwidth, 25).astype('uint'))
        plt.yticks((img.shape[0] - np.arange(25, mapheight, 25) * mapscaler).astype('uint'), np.arange(25, mapheight, 25).astype('uint'))
        plt.show()


class DLModel:
    def __init__(self, site, floor, batchsize=64, testratio=0.1, device='cuda', use_augmentation=True, use_wifi=True, use_ibeacon=True):
        self.site, self.floor = site, floor
        self.batchsize = batchsize
        self.testratio = testratio
        self.use_augmentation = use_augmentation
        self.trained_epochs = 0
        self.loss_error = [[],[],[]] # trainloss, trainerror, testerror
        self.trainDataLoader, self.testDataLoader = None, None
        self.y_mean, self.y_std = None, None
        self.lr = None
        self.model = None
        self.bssid2index = None
        self.uuid2index = None 
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_wifi=use_wifi
        self.use_ibeacon=use_ibeacon
        self.scaler=None

    def initial(self):
        self.load_data()
        self.initial_model(len(self.bssid2index), len(self.uuid2index))

    def load_data(self): # create a dataloader for train and test
        print('Loading and preprocessing data from txt...')
        train_set, test_set, (self.bssid2index, self.uuid2index) = split_floor_data(self.site, self.floor, self.testratio, augmentation=self.use_augmentation)
        scaler = StandardScaler()
        self.scaler=scaler
        train_set[:,2:6] = scaler.fit_transform(train_set[:,2:6].copy())
        test_set[:,2:6] = scaler.transform(test_set[:,2:6].copy())
        y = train_set[:, :2]
        self.y_mean, self.y_std = torch.Tensor(y.mean(axis=0)).to(self.device), torch.Tensor(y.std(axis=0)).to(self.device)
        self.trainDataLoader = DataLoader(FloorData(train_set), batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(FloorData(test_set), batch_size=self.batchsize, shuffle=False)
        print('data loading finish')

    def initial_model(self, wifi_dim, ibeacon_dim):
        self.model = DLnetwork(wifi_dim, ibeacon_dim, augmentation=self.use_augmentation, 
            use_wifi=self.use_wifi, use_ibeacon=self.use_ibeacon).to(self.device)
        print('model initialization finish')

    def train(self, epochs, reduce_lr_epoch=5, startlr=0.1, verbose=True):
        if startlr is None:
            if self.lr is None:
                cur_lr = 0.01
            else:
                cur_lr = self.lr 
        else:
            cur_lr = startlr
            self.lr = cur_lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cur_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, 
                                                               patience=reduce_lr_epoch, cooldown=5, min_lr=1e-5)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(1, epochs+1):
            epoch_loss = 0
            epoch_error = 0
            batch_number = 0

            for x, y in iter(self.trainDataLoader):
                batch_number += 1 
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                pred = self.model(x)
                error = torch.sum(torch.sqrt(torch.sum((y-(pred*self.y_std + self.y_mean))**2, dim=1))) / y.shape[0]
                epoch_error += error.detach().item()

                loss = criterion(pred, (y-self.y_mean)/self.y_std)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

            mean_loss = epoch_loss / batch_number
            mean_error = epoch_error / batch_number
            test_error = self.evaluate()
            self.trained_epochs += 1
            self.loss_error[0].append(mean_loss)
            self.loss_error[1].append(mean_error)
            self.loss_error[2].append(test_error.detach().item())
            if verbose:
              print(f'epoch: {self.trained_epochs} | lr: {optimizer.param_groups[0]["lr"]} |epoch loss: {mean_loss} | \
                  Euclidean distance error on epoch: {mean_error}| Euclidean distance error on TestSet: {test_error} ')   
            #scheduler.step(mean_loss)
            scheduler.step(test_error.detach())
            
            if epoch > 100 and all(past_error <= test_error.detach().item() for past_error in list(self.loss_error[2])[-20:]):
              print(f'Early stopping at Epoch {epoch}')
              break

        return np.min(self.loss_error[2]), np.argmin(self.loss_error[2])+1



    def evaluate(self):
        self.model.eval()
        total_error = 0
        batch_number = 0
        with torch.no_grad():
            for x, y in iter(self.testDataLoader):
                batch_number += 1 
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                pred = self.model(x)
                error = torch.sum(torch.sqrt(torch.sum((y-(pred*self.y_std + self.y_mean))**2, dim=1))) / y.shape[0]
                total_error += error
        mean_error = total_error / batch_number
        return mean_error


    def plotloss(self):
        epochs = list(range(1, len(self.loss_error[0])+1))
        if not epochs:
            print(f'0 epoch trained')
            return

        plt.clf()
        fig = plt.figure(figsize=(12,8), dpi=80)
        loss_fig = fig.add_subplot(2,1,1)
        error_fig = fig.add_subplot(2,1,2)
        loss_fig.plot(epochs, self.loss_error[0], label='train loss',linestyle='-', color='red')
        error_fig.plot(epochs, self.loss_error[1], label='train error', linestyle='-', color='blue')
        error_fig.plot(epochs, self.loss_error[2], label='test error', linestyle='-', color='green')
        loss_fig.legend(loc='upper right')
        error_fig.legend(loc='upper right')
        plt.savefig('loss.jpg', bbox_inches='tight')
        plt.show()


    def predict(self, data, groundtruth=None):
        # data = {Mx, My, Mz, {bssid:rssi}, {uuid:rssi}}
        self.model.eval()
        x = torch.zeros([1,6+len(self.bssid2index)+len(self.uuid2index)])
        Mx, My, Mz = data[:3]
        MI = (Mx**2 + My**2 + Mz**2)**0.5
        Mx, My, Mz, MI = self.scaler.transform(np.array([[Mx,My,Mz,MI]]))[0]
        wifis = data[3]
        ibeacons = data[4]
        wifi_det, ibeacon_det = 1 if wifis else 0, 1 if ibeacons else 0
        x[0][:6] = torch.Tensor([[Mx, My, Mz, MI, wifi_det, ibeacon_det]])
        for bssid, rssi in wifis.items():
            if bssid in self.bssid2index:
                x[0][6+self.bssid2index[bssid]] = (100+rssi) / 100 
        for uuid, rssi in ibeacons.items():
            if uuid in self.uuid2index:
                x[0][6+len(self.bssid2index)+self.uuid2index[uuid]] = (100+rssi) / 100
        x = x.to(self.device)

        pred = self.model(x)
        pred = pred * self.y_std + self.y_mean
        pred = (float(pred[0][0]), float(pred[0][1]))

        plt.clf()
        plt.figure(figsize=(6,6), dpi=160)
        json_path = os.path.join('../data', self.site, self.floor, 'floor_info.json')
        with open(json_path) as file:
            mapinfo = json.load(file)['map_info']
        mapheight, mapwidth = mapinfo['height'], mapinfo['width']
        img = mpimg.imread(os.path.join('../data', self.site, self.floor, 'floor_image.png'))
        plt.imshow(img)
        mapscaler = (img.shape[0]/mapheight + img.shape[1]/mapwidth)/2
        pre = plt.scatter([pred[0]*mapscaler], [img.shape[0] - pred[1]*mapscaler], color='red', marker='o', s=7)
        if groundtruth:
            real = plt.scatter([groundtruth[0]*mapscaler], [img.shape[0] - groundtruth[1]*mapscaler], color='green', marker='x', s=7)
            plt.legend([pre, real], ['prediction', 'groundtruth'], loc='lower left')
        else:
            plt.legend([pre], ['prediction'], loc='lower left')
        plt.xticks((np.arange(25, mapwidth, 25) * mapscaler).astype('uint'), np.arange(25, mapwidth, 25).astype('uint'))
        plt.yticks((img.shape[0] - np.arange(25, mapheight, 25) * mapscaler).astype('uint'), np.arange(25, mapheight, 25).astype('uint'))
        plt.show()


def get_data_from_one_txt(txtpath, augmentation=True):
    acce = []
    magn = []
    ahrs = []
    wifi = []
    ibeacon = []
    waypoint = []

    with open(txtpath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line or line[0] == '#':
                continue

            line_data = line.split('\t')

            if line_data[1] == 'TYPE_ACCELEROMETER':
                acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            elif line_data[1] == 'TYPE_MAGNETIC_FIELD':
                magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            elif line_data[1] == 'TYPE_ROTATION_VECTOR':
                ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            elif line_data[1] == 'TYPE_WIFI':
                sys_ts = line_data[0]
                bssid = line_data[3]
                rssi = line_data[4]
                wifi_data = [sys_ts, bssid, rssi]
                wifi.append(wifi_data)
            elif line_data[1] == 'TYPE_BEACON':
                ts = line_data[0]
                uuid = line_data[2]
                major = line_data[3]
                minor = line_data[4]
                rssi = line_data[6]
                ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
                ibeacon.append(ibeacon_data)
            elif line_data[1] == 'TYPE_WAYPOINT':
                waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

    acce, magn, ahrs, wifi, ibeacon, waypoint = np.array(acce), np.array(magn), np.array(ahrs), np.array(wifi), np.array(ibeacon), np.array(waypoint)
    
    if augmentation:
        augmented_data = Compute().compute_step_positions(acce, ahrs, waypoint) # use position estimation funciton in sample code compute_f.py
    else:
        augmented_data = waypoint

    index2data = [{'magn':[], 'wifi':defaultdict(list), 'ibeacon':defaultdict(list)} for _ in range(len(augmented_data))]
    index2time = augmented_data[:,0]
    for magn_data in magn: 
        tdiff = abs(index2time - magn_data[0])
        i = np.argmin(tdiff)
        index2data[i]['magn'].append((magn_data[1], magn_data[2], magn_data[3])) # 'magn': [(x1,y1,z1), (x2,y2,z2),...]
    for wifi_data in wifi:
        tdiff = abs(index2time - int(wifi_data[0]))
        i = np.argmin(tdiff)
        index2data[i]['wifi'][wifi_data[1]].append(int(wifi_data[2])) # 'wifi': {'id1':[-50, -20], 'id7':[-10]}
    for ibeacon_data in ibeacon:
        tdiff = abs(index2time - int(ibeacon_data[0]))
        i = np.argmin(tdiff)
        index2data[i]['ibeacon'][ibeacon_data[1]].append(int(ibeacon_data[2])) # 'wifi': {'id2':[-50, -24], 'id5':[-12,-30,-49]}

    txt_data = [None] * len(augmented_data)
    for index in range(len(index2time)):
        t, Px, Py = augmented_data[index]
        txt_data[index] = [t,Px,Py]
        magns, wifis, ibeacons = np.array(index2data[index]['magn']), index2data[index]['wifi'], index2data[index]['ibeacon'] 
        if len(magns) > 0:
            magn_mean = magns.mean(axis=0)
            magn_mean_intense = np.mean(np.sqrt(np.sum(magns**2, axis=1)))
            txt_data[index].extend(list(magn_mean) + [float(magn_mean_intense)])
        else:
            txt_data[index].extend([0,0,0,0])

        txt_data[index].append(defaultdict(lambda: -100))
        for bssid, rssis in wifis.items():
            txt_data[index][-1][bssid] = sum(rssis)/len(rssis)

        txt_data[index].append(defaultdict(lambda: -100))
        for uuid, rssis in ibeacons.items():
            txt_data[index][-1][uuid] = sum(rssis)/len(rssis)

    # returned format [(time, POSx, POSy, magnX, magnY, magnZ, magnIntense, {'BSSID4':rssi, 'BSSID7':rssi,..}, {'UUID2':rssi, 'UUID7':rssi,..}),...]
    return txt_data 

def split_floor_data(site, floor, testratio=0.1, augmentation=True): # (100 + rssi) / 100  ->  (0,1)
    file_path = os.path.join(os.path.join(os.getcwd(),'data'), site, floor)
    file_list = os.listdir(os.path.join(file_path, "path_data_files"))

    file_list = [file_list[0]]
    print(file_list[0])

    total_posMagn_data = np.zeros((0, 6)).astype('float') # (Posx, Posy, MagnX, MagnY, MagnZ, MagnI)
    total_wifi_data = np.zeros((0,0)).astype('float') 
    total_ibeacon_data = np.zeros((0,0)).astype('float') 
    wifi_ibeacon_detected = np.zeros((0,2)).astype('float') # 记录这个时间点是否有对wifi和ibeacon有检测，没有记为0
    index2bssid = []
    bssid2index = dict()
    index2uuid = []
    uuid2index = dict()
    no_wifi_ibeacon = [0,0]
    not_in_train_wifi_ibeacon = [0,0]

    trajectory_data = np.zeros((0,9))
    curfilenum = 0 # Del
    for filename in file_list:
        curfilenum += 1 # Del
        if curfilenum % 10 == 0: # Del
            print(f'already read {curfilenum} txts') # Del
        txtname = os.path.join(file_path, "path_data_files", filename)
        _ = np.array(get_data_from_one_txt(txtname, augmentation=augmentation))
        #print(_.shape)
        #print(_[0][7].items())
        trajectory_data = np.append(trajectory_data, _, axis=0)
    
    total_posMagn_data = trajectory_data[:, 1:7].astype('float')
    data_number = total_posMagn_data.shape[0]
    test_number = int(testratio * data_number)
    train_number = data_number - test_number
    test_indices = random.sample(range(data_number), test_number)
    train_indices = list(set(range(data_number)).difference(test_indices))
    finish_number = 0

    # add train data the total data
    for index in train_indices:
        # add one instance to total_data
        finish_number += 1 
        if finish_number % 500 == 0:
            print(f'data processing ... {finish_number}/{data_number}')
        tdata = trajectory_data[index]
        wifi_ibeacon_detected = np.concatenate((wifi_ibeacon_detected, np.zeros((1, 2))), axis=0)
        total_wifi_data = np.concatenate((total_wifi_data, np.zeros((1,total_wifi_data.shape[1]))), axis=0)
        total_ibeacon_data = np.concatenate((total_ibeacon_data, np.zeros((1,total_ibeacon_data.shape[1]))), axis=0)

        wifidic = tdata[7]
        if wifidic:
            wifi_ibeacon_detected[-1][0] = 1
        else: 
            no_wifi_ibeacon[0] += 1 
        for bssid, rssi in wifidic.items():
            if bssid not in bssid2index: # for train set, if a bssid did not appear before, we should add it to a new feature.
                bssid2index[bssid] = len(index2bssid)
                index2bssid.append(bssid)
                total_wifi_data = np.concatenate((total_wifi_data, np.zeros((total_wifi_data.shape[0], 1))), axis=1) # add a new feature
            total_wifi_data[-1][bssid2index[bssid]] = (100 + rssi) / 100

        ibeacondic = tdata[8]
        if ibeacondic:
            wifi_ibeacon_detected[-1][1] = 1
        else: 
            no_wifi_ibeacon[1] += 1 
        for uuid, rssi in ibeacondic.items():
            if uuid not in uuid2index: # for train set, if a uuid did not appear before, we should add it to a new feature.
                uuid2index[uuid] = len(index2uuid)
                index2uuid.append(uuid)
                total_ibeacon_data = np.concatenate((total_ibeacon_data, np.zeros((total_ibeacon_data.shape[0], 1))), axis=1) # new feature
            total_ibeacon_data[-1][uuid2index[uuid]] = (100 + rssi) / 100

    # add test data the total data
    for index in test_indices:
        # add one instance to total_data
        finish_number += 1 
        if finish_number % 500 == 0:
            print(f'data processing ... {finish_number}/{data_number}')
        tdata = trajectory_data[index]
        wifi_ibeacon_detected = np.concatenate((wifi_ibeacon_detected, np.zeros((1, 2))), axis=0)
        total_wifi_data = np.concatenate((total_wifi_data, np.zeros((1,total_wifi_data.shape[1]))), axis=0)
        total_ibeacon_data = np.concatenate((total_ibeacon_data, np.zeros((1,total_ibeacon_data.shape[1]))), axis=0)

        wifidic = tdata[7]
        if wifidic:
            wifi_ibeacon_detected[-1][0] = 1
        else: 
            no_wifi_ibeacon[0] += 1 
        for bssid, rssi in wifidic.items():
            if bssid in bssid2index: # For test data, we only caputure the bssid which appeared in the train data before.
                total_wifi_data[-1][bssid2index[bssid]] = (100 + rssi) / 100
            else:
                not_in_train_wifi_ibeacon[0] += 1


        ibeacondic = tdata[8]
        if ibeacondic:
            wifi_ibeacon_detected[-1][1] = 1
        else:
            no_wifi_ibeacon[1] += 1
        for uuid, rssi in ibeacondic.items():
            if uuid in uuid2index: # For test data, we only caputure the bssid which appeared in the train data before.
                total_ibeacon_data[-1][uuid2index[uuid]] = (100 + rssi) / 100
            else:
                not_in_train_wifi_ibeacon[1] += 1

    df_total_posMagn_data = pd.DataFrame(total_posMagn_data)
    df_total_posMagn_data.to_csv(os.path.join(os.path.join(os.getcwd(), 'output'), 'total_posMagn_data.csv'))

    df_wifi_ibeacon_detected = pd.DataFrame(wifi_ibeacon_detected)
    df_wifi_ibeacon_detected.to_csv(os.path.join(os.path.join(os.getcwd(), 'output'), 'wifi_ibeacon_detected.csv'))

    df_total_wifi_data = pd.DataFrame(total_wifi_data)
    df_total_wifi_data.to_csv(os.path.join(os.path.join(os.getcwd(), 'output'), 'total_wifi_data.csv'))

    df_total_ibeacon_data = pd.DataFrame(total_ibeacon_data)
    df_total_ibeacon_data.to_csv(os.path.join(os.path.join(os.getcwd(), 'output'), 'total_ibeacon_data.csv'))

    train_set = np.concatenate((total_posMagn_data[train_indices, :], wifi_ibeacon_detected[:train_number, :], \
                                total_wifi_data[:train_number, :], total_ibeacon_data[:train_number, :]), axis=1)
    test_set = np.concatenate((total_posMagn_data[test_indices, :], wifi_ibeacon_detected[train_number:, :], \
                                total_wifi_data[train_number:, :], total_ibeacon_data[train_number:, :]), axis=1)

    print(f'Total data instance: {data_number}, train number: {train_number}, test number: {test_number}')
    print()    
    print(f'There are {no_wifi_ibeacon[0]} steps that did not find wifi, and {no_wifi_ibeacon[1]} steps that did not find ibeacon.')
    print(f'There are {not_in_train_wifi_ibeacon[0]} bssids and {not_in_train_wifi_ibeacon[1]} uuids which were detected in testset while not detected in trainset, so that they were discard.')
    print(f'Final wifi bssid number: {len(index2bssid)}, ibeacon uuid number: {len(index2uuid)}')

    return train_set, test_set, [bssid2index, uuid2index]