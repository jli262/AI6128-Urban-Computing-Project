import os
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List
import json
import plotly.graph_objs as go
from PIL import Image
import scipy.signal as signal

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
    
    def load_info(self):
        path = os.path.join(
                    os.path.join(
                        os.path.join(self.directory, self.site
                            ), self.floor
                        ), 'floor_info.json'
                    )
        with open(path, 'r') as f:
            self.info = json.load(f)['map_info']
    
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
    


@dataclass
class ReadData:
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
        
    def load(self):
        with open(self.filepath, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()
    
    def parse(self):
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

        return ReadData(self.acce, self.acce_uncali, self.gyro, self.gyro_uncali, self.magn, self.magn_uncali, self.ahrs, self.wifi, self.ibeacon, self.waypoint)


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
    
