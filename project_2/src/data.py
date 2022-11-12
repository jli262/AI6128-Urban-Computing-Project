import logging
logger = logging.getLogger(__name__)

import sys
import os
import pandas as pd
import json
import osmnx as ox
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt

class Trip(object):
    def __init__(self, named_tuple):
        fields = named_tuple._fields
        i = 0
        for val in named_tuple:
            setattr(self, fields[i], val)
            i+=1

class Trips(object):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.trips = []

    def load(self):
        if self.filepath != '':
            data = pd.read_csv(self.filepath)
        
        for row in data.itertuples():
            trip = Trip(row)
            self.trips.append(trip)
                
class RoadNetwork(object):
    def __init__(self):
        self.graph = None
        self.fig = None
        self.ax = None
    
    def from_polygon(self, filepath: str):
        geojson = GeoJson(filepath)
        geojson.load()
        polygon = geojson.boundary_polygon
        self.graph = ox.graph_from_polygon(polygon, network_type='drive')

    def plot_graph(self, node_size=3, figsize=(20,20), edge_linewidth=1.5, show=False):
        if not self.graph is None:
            self.fig, self.ax = ox.plot_graph(self.graph, node_size=node_size, 
                                                figsize=figsize, edge_linewidth=edge_linewidth, 
                                                show=show)
            
    def plot_trip(self, trip: Trip, as_line: bool = False, color='red', show: bool=False):
        if (not self.fig is None) and (not self.ax is None):
            points = eval(trip.POLYLINE)
            x, y = zip(*points)
            x_max, x_min, x_gap, x_mid = max(x), min(x), max(x) - min(x), (max(x) + min(x))/2
            y_max, y_min, y_gap, y_mid = max(y), min(y), max(y) - min(y), (max(y) + min(y))/2
            gap = max(x_gap, y_gap) * 0.6
            self.ax.set_xlim(x_mid - gap, x_mid + gap)
            self.ax.set_ylim(y_mid - gap, y_mid + gap)

            if as_line:
                self.ax.plot(x, y, linewidth = 4, color=color, linestyle='-', marker='x', markersize=20)
            else:
                self.ax.scatter(x, y, c=color, marker='x',s=20)
                        
        if show:
            plt.show()

    def save_plot(self, filepath: str, dpi: int=80):
        self.fig.savefig(filepath, dpi=dpi)

    def to_shapefile_directional(self, directory, encoding="utf-8"):
        # default filepath if none was provided
        #if filepath is None:
        #    filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

        # if save folder does not already exist, create it (shapefiles
        # get saved as set of files)
        if not directory == "" and not os.path.exists(directory):
            os.makedirs(directory)
        filepath_nodes = os.path.join(directory, "nodes.shp")
        filepath_edges = os.path.join(directory, "edges.shp")

        # convert undirected graph to gdfs and stringify non-numeric columns
        gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(self.graph)
        gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
        gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)

        # We need an unique ID for each edge
        gdf_edges["fid"] = gdf_edges.index.map('_'.join)
        
        # save the nodes and edges as separate ESRI shapefiles
        gdf_nodes.to_file(filepath_nodes, encoding=encoding)
        gdf_edges.to_file(filepath_edges, encoding=encoding)


class GeoJson(object):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load(self):
        if self.filepath != '':
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError('`filepath` missing...')

    @property
    def boundary_polygon(self):
        if not self.data is None:
            return shape(self.data['geometries'][0])
    


if __name__ == '__main__':
    cwd = os.getcwd()
    data_directory = os.path.join(os.path.dirname(cwd), 'data')
    data_filepath = os.path.join(data_directory, 'train.csv')

    df = pd.read_csv(data_filepath)
    print(df.head())
