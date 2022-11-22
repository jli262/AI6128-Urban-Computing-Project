import logging
logger = logging.getLogger(__name__)

import os
from src.data import Trips, RoadNetwork
import matplotlib.colors as mcolors
import osmnx
import folium

if __name__ == '__main__':
    datapath = os.path.join(os.path.join(os.getcwd(), 'data'), 'train_1000.csv')
    jsonpath = os.path.join(os.path.join(os.getcwd(), 'data'), 'porto.geojson')
    output_directory = os.path.join(os.getcwd(), 'output')
    trips = Trips(datapath)
    trips.load()

    road_network = RoadNetwork()
    road_network.from_polygon(jsonpath)
    road_network.plot_graph()
    
    first_ten = trips.trips[:10]
    road_network.to_shapefile_directional(output_directory)
    
    i = 1
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for trip in first_ten:
        uid = 'trip_{num}.png'.format(num=i)
        road_network.plot_trip(trip, color='red')
        road_network.save_plot(os.path.join(output_directory, uid))
        road_network.plot_graph()
        i+=1

    i = 1
    color_list = ['red',  'orange', 'yellow',  'olive', 'green', 'blue', 'purple', 'cyan',  'pink', 'black']
    for trip in first_ten:
        uid = 'trip_{num}.html'.format(num=i)
        G = road_network.graph
        route = eval(trip.POLYLINE)
        route = [(lati, longi) for longi, lati in route]
        map_ = osmnx.folium.plot_graph_folium(G, color='#D3D3D3', weight = 1, zoom = 14, zoom_control = False)
        folium.PolyLine(route, color=color_list[i-1], weight=7, opacity=0.8).add_to(map_)
        map_.save(outfile=os.path.join(output_directory, uid))
        i+=1



    