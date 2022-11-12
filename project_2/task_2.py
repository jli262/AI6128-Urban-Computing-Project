import logging
logger = logging.getLogger(__name__)

import os
from src.data import Trips, RoadNetwork
import matplotlib.colors as mcolors

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
    i = 1
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for trip in first_ten:
        uid = 'trip_{num}.png'.format(num=i)
        road_network.plot_trip(trip, color='red')
        road_network.save_plot(os.path.join(output_directory, uid))
        road_network.plot_graph()
        i+=1



    