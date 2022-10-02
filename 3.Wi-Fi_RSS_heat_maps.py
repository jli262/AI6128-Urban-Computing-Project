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

class Compute:
    def split_ts_seq(self,ts_seq, sep_ts):
        """

        :param ts_seq:
        :param sep_ts:
        :return:
        """
        tss = ts_seq[:, 0].astype(float)
        unique_sep_ts = np.unique(sep_ts)
        ts_seqs = []
        start_index = 0
        for i in range(0, unique_sep_ts.shape[0]):
            end_index = np.searchsorted(tss, unique_sep_ts[i], side='right')
            if start_index == end_index:
                continue
            ts_seqs.append(ts_seq[start_index:end_index, :].copy())
            start_index = end_index

        # tail data
        if start_index < ts_seq.shape[0]:
            ts_seqs.append(ts_seq[start_index:, :].copy())

        return ts_seqs


    def correct_trajectory(self,original_xys, end_xy):
        """

        :param original_xys: numpy ndarray, shape(N, 2)
        :param end_xy: numpy ndarray, shape(1, 2)
        :return:
        """
        corrected_xys = np.zeros((0, 2))

        A = original_xys[0, :]
        B = end_xy
        Bp = original_xys[-1, :]

        angle_BAX = np.arctan2(B[1] - A[1], B[0] - A[0])
        angle_BpAX = np.arctan2(Bp[1] - A[1], Bp[0] - A[0])
        angle_BpAB = angle_BpAX - angle_BAX
        AB = np.sqrt(np.sum((B - A) ** 2))
        ABp = np.sqrt(np.sum((Bp - A) ** 2))

        corrected_xys = np.append(corrected_xys, [A], 0)
        for i in np.arange(1, np.size(original_xys, 0)):
            angle_CpAX = np.arctan2(original_xys[i, 1] - A[1], original_xys[i, 0] - A[0])

            angle_CAX = angle_CpAX - angle_BpAB

            ACp = np.sqrt(np.sum((original_xys[i, :] - A) ** 2))

            AC = ACp * AB / ABp

            delta_C = np.array([AC * np.cos(angle_CAX), AC * np.sin(angle_CAX)])

            C = delta_C + A

            corrected_xys = np.append(corrected_xys, [C], 0)

        return corrected_xys


    def correct_positions(self, rel_positions, reference_positions):
        """

        :param rel_positions:
        :param reference_positions:
        :return:
        """
        rel_positions_list = self.split_ts_seq(rel_positions, reference_positions[:, 0])
        if len(rel_positions_list) != reference_positions.shape[0] - 1:
            # print(f'Rel positions list size: {len(rel_positions_list)}, ref positions size: {reference_positions.shape[0]}')
            del rel_positions_list[-1]
        assert len(rel_positions_list) == reference_positions.shape[0] - 1

        corrected_positions = np.zeros((0, 3))
        for i, rel_ps in enumerate(rel_positions_list):
            start_position = reference_positions[i]
            end_position = reference_positions[i + 1]
            abs_ps = np.zeros(rel_ps.shape)
            abs_ps[:, 0] = rel_ps[:, 0]
            # abs_ps[:, 1:3] = rel_ps[:, 1:3] + start_position[1:3]
            abs_ps[0, 1:3] = rel_ps[0, 1:3] + start_position[1:3]
            for j in range(1, rel_ps.shape[0]):
                abs_ps[j, 1:3] = abs_ps[j-1, 1:3] + rel_ps[j, 1:3]
            abs_ps = np.insert(abs_ps, 0, start_position, axis=0)
            corrected_xys = self.correct_trajectory(abs_ps[:, 1:3], end_position[1:3])
            corrected_ps = np.column_stack((abs_ps[:, 0], corrected_xys))
            if i == 0:
                corrected_positions = np.append(corrected_positions, corrected_ps, axis=0)
            else:
                corrected_positions = np.append(corrected_positions, corrected_ps[1:], axis=0)

        corrected_positions = np.array(corrected_positions)

        return corrected_positions


    def init_parameters_filter(self,sample_freq, warmup_data, cut_off_freq=2):
        order = 4
        filter_b, filter_a = signal.butter(order, cut_off_freq / (sample_freq / 2), 'low', False)
        zf = signal.lfilter_zi(filter_b, filter_a)
        _, zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)
        _, filter_zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)

        return filter_b, filter_a, filter_zf


    def get_rotation_matrix_from_vector(self,rotation_vector):
        q1 = rotation_vector[0]
        q2 = rotation_vector[1]
        q3 = rotation_vector[2]

        if rotation_vector.size >= 4:
            q0 = rotation_vector[3]
        else:
            q0 = 1 - q1*q1 - q2*q2 - q3*q3
            if q0 > 0:
                q0 = np.sqrt(q0)
            else:
                q0 = 0

        sq_q1 = 2 * q1 * q1
        sq_q2 = 2 * q2 * q2
        sq_q3 = 2 * q3 * q3
        q1_q2 = 2 * q1 * q2
        q3_q0 = 2 * q3 * q0
        q1_q3 = 2 * q1 * q3
        q2_q0 = 2 * q2 * q0
        q2_q3 = 2 * q2 * q3
        q1_q0 = 2 * q1 * q0

        R = np.zeros((9,))
        if R.size == 9:
            R[0] = 1 - sq_q2 - sq_q3
            R[1] = q1_q2 - q3_q0
            R[2] = q1_q3 + q2_q0

            R[3] = q1_q2 + q3_q0
            R[4] = 1 - sq_q1 - sq_q3
            R[5] = q2_q3 - q1_q0

            R[6] = q1_q3 - q2_q0
            R[7] = q2_q3 + q1_q0
            R[8] = 1 - sq_q1 - sq_q2

            R = np.reshape(R, (3, 3))
        elif R.size == 16:
            R[0] = 1 - sq_q2 - sq_q3
            R[1] = q1_q2 - q3_q0
            R[2] = q1_q3 + q2_q0
            R[3] = 0.0

            R[4] = q1_q2 + q3_q0
            R[5] = 1 - sq_q1 - sq_q3
            R[6] = q2_q3 - q1_q0
            R[7] = 0.0

            R[8] = q1_q3 - q2_q0
            R[9] = q2_q3 + q1_q0
            R[10] = 1 - sq_q1 - sq_q2
            R[11] = 0.0

            R[12] = R[13] = R[14] = 0.0
            R[15] = 1.0

            R = np.reshape(R, (4, 4))

        return R


    def get_orientation(self,R):
        flat_R = R.flatten()
        values = np.zeros((3,))
        if np.size(flat_R) == 9:
            values[0] = np.arctan2(flat_R[1], flat_R[4])
            values[1] = np.arcsin(-flat_R[7])
            values[2] = np.arctan2(-flat_R[6], flat_R[8])
        else:
            values[0] = np.arctan2(flat_R[1], flat_R[5])
            values[1] = np.arcsin(-flat_R[9])
            values[2] = np.arctan2(-flat_R[8], flat_R[10])

        return values


    def compute_steps(self, acce_datas):
        step_timestamps = np.array([])
        step_indexs = np.array([], dtype=int)
        step_acce_max_mins = np.zeros((0, 4))
        sample_freq = 50
        window_size = 22
        low_acce_mag = 0.6
        step_criterion = 1
        interval_threshold = 250

        acce_max = np.zeros((2,))
        acce_min = np.zeros((2,))
        acce_binarys = np.zeros((window_size,), dtype=int)
        acce_mag_pre = 0
        state_flag = 0

        warmup_data = np.ones((window_size,)) * 9.81
        filter_b, filter_a, filter_zf = self.init_parameters_filter(sample_freq, warmup_data)
        acce_mag_window = np.zeros((window_size, 1))

        # detect steps according to acceleration magnitudes
        for i in np.arange(0, np.size(acce_datas, 0)):
            acce_data = acce_datas[i, :]
            acce_mag = np.sqrt(np.sum(acce_data[1:] ** 2))

            acce_mag_filt, filter_zf = signal.lfilter(filter_b, filter_a, [acce_mag], zi=filter_zf)
            acce_mag_filt = acce_mag_filt[0]

            acce_mag_window = np.append(acce_mag_window, [acce_mag_filt])
            acce_mag_window = np.delete(acce_mag_window, 0)
            mean_gravity = np.mean(acce_mag_window)
            acce_std = np.std(acce_mag_window)
            mag_threshold = np.max([low_acce_mag, 0.4 * acce_std])

            # detect valid peak or valley of acceleration magnitudes
            acce_mag_filt_detrend = acce_mag_filt - mean_gravity
            if acce_mag_filt_detrend > np.max([acce_mag_pre, mag_threshold]):
                # peak
                acce_binarys = np.append(acce_binarys, [1])
                acce_binarys = np.delete(acce_binarys, 0)
            elif acce_mag_filt_detrend < np.min([acce_mag_pre, -mag_threshold]):
                # valley
                acce_binarys = np.append(acce_binarys, [-1])
                acce_binarys = np.delete(acce_binarys, 0)
            else:
                # between peak and valley
                acce_binarys = np.append(acce_binarys, [0])
                acce_binarys = np.delete(acce_binarys, 0)

            if (acce_binarys[-1] == 0) and (acce_binarys[-2] == 1):
                if state_flag == 0:
                    acce_max[:] = acce_data[0], acce_mag_filt
                    state_flag = 1
                elif (state_flag == 1) and ((acce_data[0] - acce_max[0]) <= interval_threshold) and (
                        acce_mag_filt > acce_max[1]):
                    acce_max[:] = acce_data[0], acce_mag_filt
                elif (state_flag == 2) and ((acce_data[0] - acce_max[0]) > interval_threshold):
                    acce_max[:] = acce_data[0], acce_mag_filt
                    state_flag = 1

            # choose reasonable step criterion and check if there is a valid step
            # save step acceleration data: step_acce_max_mins = [timestamp, max, min, variance]
            step_flag = False
            if step_criterion == 2:
                if (acce_binarys[-1] == -1) and ((acce_binarys[-2] == 1) or (acce_binarys[-2] == 0)):
                    step_flag = True
            elif step_criterion == 3:
                if (acce_binarys[-1] == -1) and (acce_binarys[-2] == 0) and (np.sum(acce_binarys[:-2]) > 1):
                    step_flag = True
            else:
                if (acce_binarys[-1] == 0) and acce_binarys[-2] == -1:
                    if (state_flag == 1) and ((acce_data[0] - acce_min[0]) > interval_threshold):
                        acce_min[:] = acce_data[0], acce_mag_filt
                        state_flag = 2
                        step_flag = True
                    elif (state_flag == 2) and ((acce_data[0] - acce_min[0]) <= interval_threshold) and (
                            acce_mag_filt < acce_min[1]):
                        acce_min[:] = acce_data[0], acce_mag_filt
            if step_flag:
                step_timestamps = np.append(step_timestamps, acce_data[0])
                step_indexs = np.append(step_indexs, [i])
                step_acce_max_mins = np.append(step_acce_max_mins,
                                            [[acce_data[0], acce_max[1], acce_min[1], acce_std ** 2]], axis=0)
            acce_mag_pre = acce_mag_filt_detrend

        return step_timestamps, step_indexs, step_acce_max_mins


    def compute_stride_length(self,step_acce_max_mins):
        K = 0.4
        K_max = 0.8
        K_min = 0.4
        para_a0 = 0.21468084
        para_a1 = 0.09154517
        para_a2 = 0.02301998

        stride_lengths = np.zeros((step_acce_max_mins.shape[0], 2))
        k_real = np.zeros((step_acce_max_mins.shape[0], 2))
        step_timeperiod = np.zeros((step_acce_max_mins.shape[0] - 1, ))
        stride_lengths[:, 0] = step_acce_max_mins[:, 0]
        window_size = 2
        step_timeperiod_temp = np.zeros((0, ))

        # calculate every step period - step_timeperiod unit: second
        for i in range(0, step_timeperiod.shape[0]):
            step_timeperiod_data = (step_acce_max_mins[i + 1, 0] - step_acce_max_mins[i, 0]) / 1000
            step_timeperiod_temp = np.append(step_timeperiod_temp, [step_timeperiod_data])
            if step_timeperiod_temp.shape[0] > window_size:
                step_timeperiod_temp = np.delete(step_timeperiod_temp, [0])
            step_timeperiod[i] = np.sum(step_timeperiod_temp) / step_timeperiod_temp.shape[0]

        # calculate parameters by step period and acceleration magnitude variance
        k_real[:, 0] = step_acce_max_mins[:, 0]
        k_real[0, 1] = K
        for i in range(0, step_timeperiod.shape[0]):
            k_real[i + 1, 1] = np.max([(para_a0 + para_a1 / step_timeperiod[i] + para_a2 * step_acce_max_mins[i, 3]), K_min])
            k_real[i + 1, 1] = np.min([k_real[i + 1, 1], K_max]) * (K / K_min)

        # calculate every stride length by parameters and max and min data of acceleration magnitude
        stride_lengths[:, 1] = np.max([(step_acce_max_mins[:, 1] - step_acce_max_mins[:, 2]),
                                    np.ones((step_acce_max_mins.shape[0], ))], axis=0)**(1 / 4) * k_real[:, 1]

        return stride_lengths


    def compute_headings(self,ahrs_datas):
        headings = np.zeros((np.size(ahrs_datas, 0), 2))
        for i in np.arange(0, np.size(ahrs_datas, 0)):
            ahrs_data = ahrs_datas[i, :]
            rot_mat = self.get_rotation_matrix_from_vector(ahrs_data[1:])
            azimuth, pitch, roll = self.get_orientation(rot_mat)
            around_z = (-azimuth) % (2 * np.pi)
            headings[i, :] = ahrs_data[0], around_z
        return headings


    def compute_step_heading(self,step_timestamps, headings):
        step_headings = np.zeros((len(step_timestamps), 2))
        step_timestamps_index = 0
        for i in range(0, len(headings)):
            if step_timestamps_index < len(step_timestamps):
                if headings[i, 0] == step_timestamps[step_timestamps_index]:
                    step_headings[step_timestamps_index, :] = headings[i, :]
                    step_timestamps_index += 1
            else:
                break
        assert step_timestamps_index == len(step_timestamps)

        return step_headings


    def compute_rel_positions(self,stride_lengths, step_headings):
        rel_positions = np.zeros((stride_lengths.shape[0], 3))
        for i in range(0, stride_lengths.shape[0]):
            rel_positions[i, 0] = stride_lengths[i, 0]
            rel_positions[i, 1] = -stride_lengths[i, 1] * np.sin(step_headings[i, 1])
            rel_positions[i, 2] = stride_lengths[i, 1] * np.cos(step_headings[i, 1])

        return rel_positions


    def compute_step_positions(self, acce_datas, ahrs_datas, posi_datas):
        step_timestamps, step_indexs, step_acce_max_mins = self.compute_steps(acce_datas)
        headings = self.compute_headings(ahrs_datas)
        stride_lengths = self.compute_stride_length(step_acce_max_mins)
        step_headings = self.compute_step_heading(step_timestamps, headings)
        rel_positions = self.compute_rel_positions(stride_lengths, step_headings)
        step_positions = self.correct_positions(rel_positions, posi_datas)

        return step_positions

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
    def __init__(self, site_number: int, floor: str) -> None:
        super().__init__()
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


if __name__ == '__main__':
    floorplan = FloorPlan(1, 'F2')
   
    floorplan.plot_wifi_heatmap('dBm')

    floorplan.show()
    
