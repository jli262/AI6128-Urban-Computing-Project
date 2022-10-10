import os
import random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List
import json
import plotly.graph_objs as go
from PIL import Image
import scipy.signal as signal
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .preprocessing import Compute

class RootDirectory(object):
    _AVAILABLE_DATA = {
            'site1' : ['B1','F1','F2','F3','F4'],
            'site2' : ['B1','F1','F2','F3','F4', 'F5', 'F6', 'F7', 'F8']
        }
    def __init__(self, directory: str = 
                     r'data') -> None:
        self.directory = directory
        
    def _get_paths_from_folders(self, folders: List[str]) -> List[str]:
        files = []
        for folder in folders:
            for path in Path(folder).iterdir():
                if path.is_file():
                    files.append(path)
        return files
    
    def filepaths_all(self) -> List[str]:
        folders = []
        for site, floors in self._AVAILABLE_DATA.items():
            for floor in floors:
                folders.append(
                    os.path.join(
                        os.path.join(
                            os.path.join(self.directory, site
                                ), floor
                            ), 'path_data_files'
                        )
                    )
        return self._get_paths_from_folders(folders)
    
    def filepaths_by_site(self, site_number: int) -> List[str]:
        folders = []
        site = 'site' + str(site_number)
        path_side = os.path.join(self.directory, site)        
        if site in self._AVAILABLE_DATA:
            floors = self._AVAILABLE_DATA[site]
            for floor in floors:
                folders.append(
                    os.path.join(
                        os.path.join(path_side, floor
                            ), 'path_data_files'
                        )
                    )
            
            return self._get_paths_from_folders(folders)
        else:
            return []
        
    def filepaths_by_floor(self, floor: str) -> List[str]:
        sites = [site for site, floors in self._AVAILABLE_DATA.items() if floor in floors]
        if sites:
            folders = []
            for site in sites:
                folders.append(
                    os.path.join(
                        os.path.join(
                            os.path.join(self.directory, site                            
                                ), floor                    
                            ), 'path_data_files'
                        )
                    )
            return self._get_paths_from_folders(folders)
        else:
            return []
    
    def filepaths_by_site_and_floor(self, site_number: int, floor: str) -> List[str]:
        site = 'site' + str(site_number)
        folder = os.path.join(
                    os.path.join(
                        os.path.join(self.directory, site
                            ), floor
                        ), 'path_data_files'
                    )
        return self._get_paths_from_folders([folder])

class FloorPlan(RootDirectory):
    def __init__(self, site_number: int, floor: str,
                    data_directory: str = r'data') -> None:
        super().__init__(data_directory)
        self.site = 'site' + str(site_number)
        self.floor = floor
        self.image_path = os.path.join(
                            os.path.join(
                                os.path.join(self.directory, self.site
                                    ), self.floor
                                ), 'floor_image.png'
                            )
        self.info = dict()           
        self.figure = go.Figure()
        self.datafiles = []
        self.data = []
    
    def load_info(self):
        path = os.path.join(
                    os.path.join(
                        os.path.join(self.directory, self.site
                            ), self.floor
                        ), 'floor_info.json'
                    )
        with open(path, 'r') as f:
            self.info = json.load(f)['map_info']
    
    def load_data(self):
        paths = self.filepaths_by_site_and_floor(int(self.site[-1]), self.floor)
        self.datafiles = [DataFile(path) for path in paths]  
        data = []
        for file in self.datafiles:
            file.load()
            data.append(file.parse())
        self.data = data

    def _engineer_features(self, augment_data: bool = True):
        if not self.data:
            self.load_data()
        
        output = np.zeros((0,9))
        for datafile in self.datafiles:
            _ = datafile.engineer_features(augment_data)
            output = np.append(output, _, axis=0)

        return output

    def get_train_test_splits(self, test_size: float = 0.25,
                                augment_data: bool = True):
        data = self._engineer_features(augment_data)
        
        # train test indices
        pos_magn_data = data[:, 1:7].astype('float')
        x_count_total = pos_magn_data.shape[0]
        x_count_test = int(test_size * x_count_total)
        x_count_train = x_count_total - x_count_test
        idx_test = random.sample(range(x_count_total), x_count_test)
        idx_train = list(set(range(x_count_total)).difference(idx_test))

        # instantiating data structures
        wifi_ibeacon_exists = np.zeros((0,2)).astype('float')
        wifi_data = np.zeros((0, 0)).astype('float')
        wifi_bssid = []
        wifi_bssid_idx = dict()   
        ibeacon_uid = []
        ibeacon_uid_idx = dict()     
        ibeacon_data = np.zeros((0, 0)).astype('float')        

        # generating full dataset
        for idx in idx_train:
            _ = data[idx]

            # insert placeholders
            wifi_ibeacon_exists = np.concatenate((wifi_ibeacon_exists, np.zeros((1,2))), axis=0)
            wifi_data = np.concatenate((wifi_data, np.zeros((1, wifi_data.shape[1]))), axis=0)
            ibeacon_data = np.concatenate((ibeacon_data, np.zeros((1, ibeacon_data.shape[1]))), axis=0)

            # inserting each wifi access point rssi as a feature
            wifi_access_points = _[7]
            if wifi_access_points:
                wifi_ibeacon_exists[-1][0] = 1.0
            
            for bssid, rssi in wifi_access_points.items():
                if bssid not in wifi_bssid_idx:
                    wifi_bssid_idx[bssid] = len(wifi_bssid)
                    wifi_bssid.append(bssid)
                    wifi_data = np.concatenate((wifi_data, np.zeros((wifi_data.shape[0], 1))), axis=1)
                wifi_data[-1][wifi_bssid_idx[bssid]] = (100 + rssi) / 100

            # inserting each ibeacon access point rssi as a feature
            ibeacon_access_points = _[8]
            if ibeacon_access_points:
                wifi_ibeacon_exists[-1][1] = 1.0
            
            for uid, rssi in ibeacon_access_points.items():
                if uid not in ibeacon_uid_idx:
                    ibeacon_uid_idx[uid] = len(ibeacon_uid)
                    ibeacon_uid.append(uid)
                    ibeacon_data = np.concatenate((ibeacon_data, np.zeros((ibeacon_data.shape[0], 1))), axis=1)
                ibeacon_data[-1][ibeacon_uid_idx[uid]] = (100 + rssi) / 100
            
        for idx in idx_test:
            _ = data[idx]
            wifi_ibeacon_exists = np.concatenate((wifi_ibeacon_exists, np.zeros((1,2))), axis=0)
            wifi_data = np.concatenate((wifi_data, np.zeros((1, wifi_data.shape[1]))), axis=0)
            ibeacon_data = np.concatenate((ibeacon_data, np.zeros((1, ibeacon_data.shape[1]))), axis=0)

            wifi_access_points = _[7]
            if wifi_access_points:
                wifi_ibeacon_exists[-1][0] = 1.0

            # for test data, only check for bssid that existed in training data
            for bssid, rssi in wifi_access_points.items():  
                if bssid in wifi_bssid_idx:
                    wifi_data[-1][wifi_bssid_idx[bssid]] = (100 + rssi) / 100
                
            ibeacon_access_points = _[8]
            if ibeacon_access_points:
                wifi_ibeacon_exists[-1][1] = 1.0
            
            for uid, rssi in ibeacon_access_points.items():
                if uid in ibeacon_uid_idx:
                    ibeacon_data[-1][ibeacon_uid_idx[uid]] = (100 + rssi) / 100

        # train test split
        data_train = np.concatenate((pos_magn_data[idx_train, :],
                                        wifi_ibeacon_exists[:x_count_train, :],
                                        wifi_data[:x_count_train, :],
                                        ibeacon_data[:x_count_train, :]), axis=1)
        data_test = np.concatenate((pos_magn_data[idx_test, :],
                                        wifi_ibeacon_exists[x_count_train:, :],
                                        wifi_data[x_count_train:, :],
                                        ibeacon_data[x_count_train:, :]), axis=1)

        # standardization of magnetic field data
        scaler = StandardScaler()
        data_train[:, 2:6] = scaler.fit_transform(data_train[:, 2:6].copy())
        data_test[:, 2:6] = scaler.transform(data_test[:, 2:6].copy())        

        return data_train, data_test, [wifi_bssid_idx, ibeacon_uid_idx]

    def as_list(self):
        if not self.data:
            self.load_data()

        output = {}
        for data in self.data:
            wp = str(data.waypoint[0]) + '_' + str(data.waypoint[1])
            if wp in output:
                output[wp].append([data.id_, data.acce, data.gyro, data.magn, data.ahrs, data.wifi, data.ibeacon])
            else:
                output[wp] = [[data.id_, data.acce, data.gyro, data.magn, data.ahrs, data.wifi, data.ibeacon]]
        
        output_as_list = []
        for waypoint, v in output.items():
            for data_point in v:
                output_as_list.append(data_point + [waypoint])

        return output_as_list

    def as_dataframe(self):
        if not self.data:
            self.load_data()
        

    def show(self):
        # add floor plan        
        title = 'Floor plan for {site} {floor}'.format(site=self.site, floor=self.floor)
        if not self.info:
            self.load_info()
        floor_plan = Image.open(self.image_path)
        self.figure.update_layout(images=[
            go.layout.Image(
                source=floor_plan,
                xref="x",
                yref="y",
                x=0,
                y=self.info['height'],
                sizex=self.info['width'],
                sizey=self.info['height'],
                sizing="contain",
                opacity=1,
                layer="below",
            )
        ])
    
        # configure
        self.figure.update_xaxes(autorange=False, range=[0, self.info['width']])
        self.figure.update_yaxes(autorange=False, range=[0, self.info['height']], scaleanchor="x", scaleratio=1)
        self.figure.update_layout(
            title=go.layout.Title(
                text=title or "No title.",
                xref="paper",
                x=0,
            ),
            autosize=True,
            width=900,
            height=200 + 900 * self.info['height'] / self.info['width'],
            template="plotly_white"
        )
        
        self.figure.show()
    
    def plot_trajectory(self, mode='lines + markers + text'):
        # fig = go.Figure()
        paths = self.filepaths_by_site_and_floor(int(self.site[-1]), self.floor)
        datafiles = [DataFile(path) for path in paths]
        for file in datafiles:
            file.load()
            data = file.parse()
            trajectory = data.waypoint[:, 1:3]
    
            # add trajectory
            size_list = [6] * trajectory.shape[0]
            size_list[0] = 10
            size_list[-1] = 10
        
            color_list = ['rgba(4, 174, 4, 0.5)'] * trajectory.shape[0]
            color_list[0] = 'rgba(12, 5, 235, 1)'
            color_list[-1] = 'rgba(235, 5, 5, 1)'
        
            position_count = {}
            text_list = []
            for i in range(trajectory.shape[0]):
                if str(trajectory[i]) in position_count:
                    position_count[str(trajectory[i])] += 1
                else:
                    position_count[str(trajectory[i])] = 0
                text_list.append('        ' * position_count[str(trajectory[i])] + f'{i}')
            text_list[0] = 'Start Point: 0'
            text_list[-1] = f'End Point: {trajectory.shape[0] - 1}'
        
            self.figure.add_trace(
                go.Scattergl(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    mode=mode,
                    marker=dict(size=size_list, color=color_list),
                    line=dict(shape='linear', color='rgb(100, 10, 100)', width=2, dash='dot'),
                    text=text_list,
                    textposition="top center",
                    name='trajectory',
                ))

    def plot_magnetic(self, colorbar_title='dBm'): # Plot magnetic heatmap
        mwi_datas = self.calibrate_magnetic_wifi_ibeacon_to_position() # Calculate mwi_datas
        magnetic_strength = self.extract_magnetic_strength(mwi_datas) # Calculate magnetic_strength

        heat_positions = np.array(list(magnetic_strength.keys())) # Convert magnetic strength (K) into heat positions information
        heat_values = np.array(list(magnetic_strength.values())) #  Convert magnetic strength (Value) into heat values information
        self.figure.add_trace( # Plot heat maps
            go.Scatter(x=heat_positions[:, 0],
                       y=heat_positions[:, 1],
                       mode='markers',
                       marker=dict(size=7,
                                   color=heat_values,
                                   colorbar=dict(title=colorbar_title),
                                   colorscale="Rainbow"),
                       text=heat_values,
                       name=colorbar_title))
        pass

    def calibrate_magnetic_wifi_ibeacon_to_position(self): # Calculate mwi_datas
        mwi_datas = {}
        paths = self.filepaths_by_site_and_floor(int(self.site[-1]), self.floor) # Path call location information Site string last character
        datafiles = [DataFile(path) for path in paths]
        compute = Compute()
        for file in datafiles:
            file.load() # Load data
            data = file.parse() # Parse data
            step_positions = compute.compute_step_positions(data.acce, data.ahrs, data.waypoint)
            wifi_datas = data.wifi
            if wifi_datas.size != 0:
                sep_tss = np.unique(wifi_datas[:, 0].astype(float))
                wifi_datas_list = compute.split_ts_seq(wifi_datas, sep_tss)
                for wifi_ds in wifi_datas_list:
                    diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                    index = np.argmin(diff)
                    target_xy_key = tuple(step_positions[index, 1:3])
                    if target_xy_key in mwi_datas:
                        mwi_datas[target_xy_key]['wifi'] = np.append(mwi_datas[target_xy_key]['wifi'],
                                                                     wifi_ds, axis=0)
                    else:
                        mwi_datas[target_xy_key] = {
                            'magnetic': np.zeros((0, 4)),
                            'wifi': wifi_ds,
                            'ibeacon': np.zeros((0, 3))
                        }
            ibeacon_datas = data.ibeacon
            if ibeacon_datas.size != 0:
                sep_tss = np.unique(ibeacon_datas[:, 0].astype(float))
                ibeacon_datas_list = compute.split_ts_seq(ibeacon_datas, sep_tss)
                for ibeacon_ds in ibeacon_datas_list:
                    diff = np.abs(step_positions[:, 0] - float(ibeacon_ds[0, 0]))
                    index = np.argmin(diff)
                    target_xy_key = tuple(step_positions[index, 1:3])
                    if target_xy_key in mwi_datas:
                        mwi_datas[target_xy_key]['ibeacon'] = np.append(mwi_datas[target_xy_key]['ibeacon'], ibeacon_ds,
                                                                        axis=0)
                    else:
                        mwi_datas[target_xy_key] = {
                            'magnetic': np.zeros((0, 4)),
                            'wifi': np.zeros((0, 5)),
                            'ibeacon': ibeacon_ds
                        }

            magn_datas = data.magn
            sep_tss = np.unique(magn_datas[:, 0].astype(float))
            magn_datas_list = compute.split_ts_seq(magn_datas, sep_tss)
            for magn_ds in magn_datas_list:
                diff = np.abs(step_positions[:, 0] - float(magn_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    mwi_datas[target_xy_key]['magnetic'] = np.append(mwi_datas[target_xy_key]['magnetic'], magn_ds, axis=0)
                else:
                    mwi_datas[target_xy_key] = {
                        'magnetic': magn_ds,
                        'wifi': np.zeros((0, 5)),
                        'ibeacon': np.zeros((0, 3))
                    }
        return mwi_datas

    def extract_magnetic_strength(self, mwi_datas):
        magnetic_strength = {} # key,value
        for position_key in mwi_datas:
            # print(f'Position: {position_key}')
            magnetic_data = mwi_datas[position_key]['magnetic']
            magnetic_s = np.mean(np.sqrt(np.sum(magnetic_data[:, 1:4] ** 2, axis=1)))
            magnetic_strength[position_key] = magnetic_s
        return magnetic_strength

    def plot_wifi_heatmap(self, colorbar_title='dBm'):
        wifi_rssi = self.extract_rssi()
        ten_wifi_bssids = random.sample(wifi_rssi.keys(), 10)
        print('Example 10 wifi ap bssids:\n')
        for bssid in ten_wifi_bssids:
            print(bssid)
        target_wifi = input(f"Please input target wifi ap bssid:\n")
        heat_positions = np.array(list(wifi_rssi[target_wifi].keys()))
        print(len(heat_positions))
        heat_values = np.array(list(wifi_rssi[target_wifi].values()))[:, 0]
        # add heat map
        self.figure.add_trace(
        go.Scatter(x=heat_positions[:, 0],
                   y=heat_positions[:, 1],
                   mode='markers',
                   marker=dict(size=7,
                               color=heat_values,
                               colorbar=dict(title=colorbar_title),
                               colorscale="Rainbow"),
                   text=heat_values,
                   name=colorbar_title))

    def extract_rssi(self):
        wifi_vir_datas = {}
        paths = self.filepaths_by_site_and_floor(int(self.site[-1]), self.floor)
        datafiles = [DataFile(path) for path in paths]
        for file in datafiles:
            file.load()
            data = file.parse()
            compute = Compute()
            step_positions = compute.compute_step_positions(data.acce, data.ahrs, data.waypoint)
            wifi_datas = data.wifi
            if wifi_datas.size != 0:
                sep_tss = np.unique(wifi_datas[:, 0].astype(float))
                wifi_datas_list = compute.split_ts_seq(wifi_datas, sep_tss)
                for wifi_ds in wifi_datas_list:
                    diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                    index = np.argmin(diff)
                    target_xy_key = tuple(step_positions[index, 1:3])
                    if target_xy_key in wifi_vir_datas:
                        wifi_vir_datas[target_xy_key]['wifi'] = np.append(wifi_vir_datas[target_xy_key]['wifi'], wifi_ds, axis=0)
                    else:
                        wifi_vir_datas[target_xy_key] = {
                            'wifi': wifi_ds,
                        }

        wifi_rssi = {}
        for position_key in wifi_vir_datas:

            wifi_data = wifi_vir_datas[position_key]['wifi']
            for wifi_d in wifi_data:
                bssid = wifi_d[2]
                rssi = int(wifi_d[3])
                if bssid in wifi_rssi:
                    position_rssi = wifi_rssi[bssid]
                    if position_key in position_rssi:
                        old_rssi = position_rssi[position_key][0]
                        old_count = position_rssi[position_key][1]
                        position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                        position_rssi[position_key][1] = old_count + 1
                    else:
                        position_rssi[position_key] = np.array([rssi, 1])
                else:
                    position_rssi = {}
                    position_rssi[position_key] = np.array([rssi, 1])

                wifi_rssi[bssid] = position_rssi
        return wifi_rssi

    def plot_ibeacon_heatmap(self, title='dBm'):
        ibeacon_rssi = self.extract_ibeacon_rssi()
        print(f'This floor has {len(ibeacon_rssi.keys())} ibeacons')
        ten_ibeacon_ummids = list(ibeacon_rssi.keys())[0:10]
        print('Example 10 ibeacon UUID_MajorID_MinorIDs:\n')
        for ummid in ten_ibeacon_ummids:
            print(ummid)
        target_ibeacon = input(f"Please input target ibeacon UUID_MajorID_MinorID:\n")
        # target_ibeacon = 'FDA50693-A4E2-4FB1-AFCF-C6EB07647825_10073_61418'
        heat_positions = np.array(list(ibeacon_rssi[target_ibeacon].keys()))
        heat_values = np.array(list(ibeacon_rssi[target_ibeacon].values()))[:, 0]

        # add heatmap
        self.figure.add_trace(
            go.Scatter(x=heat_positions[:, 0],
                       y=heat_positions[:, 1],
                       mode='markers',
                       marker=dict(size=7,
                                   color=heat_values,
                                   colorbar=dict(title=title),
                                   colorscale="Rainbow"),
                       text=heat_values,
                       name=title))

    def extract_ibeacon_rssi(self):
        ibeacon_mwi_datas = {}
        paths = self.filepaths_by_site_and_floor(int(self.site[-1]), self.floor)
        datafiles = [DataFile(path) for path in paths]
        for file in datafiles:
            file.load()
            data = file.parse()
            compute = Compute()
            step_positions = compute.compute_step_positions(data.acce, data.ahrs, data.waypoint)

            ibeacon_datas = data.ibeacon

            if ibeacon_datas.size != 0:
                sep_tss = np.unique(ibeacon_datas[:, 0].astype(float))
                ibeacon_datas_list = compute.split_ts_seq(ibeacon_datas, sep_tss)
                for ibeacon_ds in ibeacon_datas_list:
                    diff = np.abs(step_positions[:, 0] - float(ibeacon_ds[0, 0]))
                    index = np.argmin(diff)
                    target_xy_key = tuple(step_positions[index, 1:3])
                    if target_xy_key in ibeacon_mwi_datas:
                        ibeacon_mwi_datas[target_xy_key]['ibeacon'] = np.append(ibeacon_mwi_datas[target_xy_key]['ibeacon'],
                                                                                ibeacon_ds, axis=0)
                    else:
                        ibeacon_mwi_datas[target_xy_key] = {
                            'ibeacon': ibeacon_ds
                        }

        ibeacon_rssi = {}
        for position_key in ibeacon_mwi_datas:
            # print(f'Position: {position_key}')

            ibeacon_data = ibeacon_mwi_datas[position_key]['ibeacon']
            for ibeacon_d in ibeacon_data:
                ummid = ibeacon_d[1]
                rssi = int(ibeacon_d[2])

                if ummid in ibeacon_rssi:
                    position_rssi = ibeacon_rssi[ummid]
                    if position_key in position_rssi:
                        old_rssi = position_rssi[position_key][0]
                        old_count = position_rssi[position_key][1]
                        position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                        position_rssi[position_key][1] = old_count + 1
                    else:
                        position_rssi[position_key] = np.array([rssi, 1])
                else:
                    position_rssi = {}
                    position_rssi[position_key] = np.array([rssi, 1])

                ibeacon_rssi[ummid] = position_rssi
        return ibeacon_rssi


class FloorData(Dataset):
    def __init__(self, data):
        self.features = data[:, 2:]
        self.labels = data[:, :2]
        self.length = data.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return self.length

@dataclass
class ReadData:
    id_: str
    acce: np.ndarray
    acce_uncali: np.ndarray
    gyro: np.ndarray
    gyro_uncali: np.ndarray
    magn: np.ndarray
    magn_uncali: np.ndarray
    ahrs: np.ndarray
    wifi: np.ndarray
    ibeacon: np.ndarray
    waypoint: np.ndarray

class DataFile(object):
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.path_id = self.filepath.name.split('.')[0]
        self.floor = self.filepath.parent.parent.name
        self.site = self.filepath.parent.parent.parent.name
        self.filepath_floorplan_image = os.path.join(self.filepath.parent.parent, 'floor_image.png')
        self.filepath_floorplan_info = os.path.join(self.filepath.parent.parent, 'floor_info.json')
        self.lines = []
        
        self.acce = []
        self.acce_uncali = []
        self.gyro = []
        self.gyro_uncali = []
        self.magn = []
        self.magn_uncali = []
        self.ahrs = []
        self.wifi = []
        self.ibeacon = []
        self.waypoint = []

        self.data = None
        self.data_with_features = None
        
    def load(self) -> None:
        with open(self.filepath, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()
    
    def parse(self) -> ReadData:
        for line_data in self.lines:
            line_data = line_data.strip()
            if not line_data or line_data[0] == '#':
                continue

            line_data = line_data.split('\t')

            if line_data[1] == 'TYPE_ACCELEROMETER':
                self.acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
                self.acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_GYROSCOPE':
                self.gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
                self.gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_MAGNETIC_FIELD':
                self.magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
                self.magn_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_ROTATION_VECTOR':
                self.ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                continue

            if line_data[1] == 'TYPE_WIFI':
                sys_ts = line_data[0]
                ssid = line_data[2]
                bssid = line_data[3]
                rssi = line_data[4]
                lastseen_ts = line_data[6]
                wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
                self.wifi.append(wifi_data)
                continue

            if line_data[1] == 'TYPE_BEACON':
                ts = line_data[0]
                uuid = line_data[2]
                major = line_data[3]
                minor = line_data[4]
                rssi = line_data[6]
                ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
                self.ibeacon.append(ibeacon_data)
                continue

            if line_data[1] == 'TYPE_WAYPOINT':
                self.waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

        self.acce = np.array(self.acce)
        self.acce_uncali = np.array(self.acce_uncali)
        self.gyro = np.array(self.gyro)
        self.gyro_uncali = np.array(self.gyro_uncali)
        self.magn = np.array(self.magn)
        self.magn_uncali = np.array(self.magn_uncali)
        self.ahrs = np.array(self.ahrs)
        self.wifi = np.array(self.wifi)
        self.ibeacon = np.array(self.ibeacon)
        self.waypoint = np.array(self.waypoint)

        self.data = ReadData(self.path_id, self.acce, self.acce_uncali, self.gyro, self.gyro_uncali, self.magn, self.magn_uncali, self.ahrs, self.wifi, self.ibeacon, self.waypoint)
        return self.data

    def engineer_features(self, augment_data: bool) -> List:
        """
            returned format [(time, POSx, POSy, magnX, magnY, magnZ, magnIntense, {'BSSID4':rssi, 'BSSID7':rssi,..}, {'UUID2':rssi, 'UUID7':rssi,..}),...]
        """
        if augment_data:
            augmented_data = Compute().compute_step_positions(self.data.acce, self.data.ahrs, self.data.waypoint) # computing steps using sample codes
        else:
            augmented_data = self.data.waypoint

        # data_index is of the following structure: List[List[Tuple[float, float, float]], Dict, Dict]
        data_index = [{'magn':[], 'wifi':defaultdict(list), 'ibeacon':defaultdict(list)} for _ in range(len(augmented_data))]
        data_timestamp = augmented_data[:,0]

        # assigning magnetic field data to waypoint with closest timestamp        
        for magn_data in self.data.magn:
            time_difference = abs(data_timestamp - int(magn_data[0]))
            idx = np.argmin(time_difference)            
            data_index[idx]['magn'].append((float(magn_data[1]), float(magn_data[2]), float(magn_data[3]))) # 'magn': [(x1,y1,z1), (x2,y2,z2), ...]

        # assigning all wifi access points rrsi to waypoint with closest timestamp
        for wifi_data in self.data.wifi:
            time_difference = abs(data_timestamp - int(wifi_data[0]))
            idx = np.argmin(time_difference)
            data_index[idx]['wifi'][wifi_data[2]].append(int(wifi_data[3]))

        # assigning all ibeacon access points rrsi to waypoint with closest timestamp
        for ibeacon_data in self.data.ibeacon:
            time_difference = abs(data_timestamp - int(ibeacon_data[0]))
            idx = np.argmin(time_difference)
            data_index[idx]['ibeacon'][ibeacon_data[1]].append(int(ibeacon_data[2]))

        # main part of feature engineering
        # 1. mean of magnetic field across all readings associated with waypoint (3 features)
        # 2. mean of magnitude of magnetic field using sqrt(a**2 + b**2 + c**2) across all readings associated with waypoint (1 feature)
        # 3. mean of rssis across each wifi access points associated with waypoint (1 feature)
        # 4. mean of rssis across each ibeacon access points associated with waypoint (1 feature)
        output = [None] * len(augmented_data)
        for idx in range(len(data_timestamp)):
            t, wp_x, wp_y = augmented_data[idx]
            output[idx] = [t, wp_x, wp_y]

            magn_data, wifi_data, ibeacon_data = (np.array(data_index[idx]['magn']),
                                                    data_index[idx]['wifi'],
                                                    data_index[idx]['ibeacon'])

            # 1. and 2. (4 features)
            if len(magn_data) > 0:
                avg = magn_data.mean(axis=0)
                magnitude = float(np.mean(np.sqrt(np.sum(magn_data**2, axis=1))))
                output[idx].extend(list(avg) + [magnitude])
            else:
                output[idx].extend([0.0, 0.0, 0.0, 0.0])

            # 3.
            output[idx].append(defaultdict(lambda: -100))
            for bssid, rssis in wifi_data.items():
                output[idx][-1][bssid] = sum(rssis) / len(rssis)
            
            # 4.
            output[idx].append(defaultdict(lambda: -100))
            for uuid, rssis in ibeacon_data.items():
                output[idx][-1][uuid] = sum(rssis) / len(rssis)
        
        self.data_with_features = output
        return self.data_with_features

class Sensor(object):
    def __init__(self):
        pass
    
class Accelerometer(Sensor):
    def __init__(self):
        pass
    
class Gyroscope(Sensor):
    def __init__(self):
        pass
    
class Magnetometer(Sensor):
    def __init__(self):
        pass

