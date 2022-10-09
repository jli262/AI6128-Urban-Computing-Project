import os
from src.data import FloorPlan

if __name__ == '__main__':
    data_directory = os.path.join(os.getcwd(), 'data')
    floorplan = FloorPlan(1, 'F1', data_directory)
    floorplan.plot_wifi_heatmap('dBm')
    floorplan.show()