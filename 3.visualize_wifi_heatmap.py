import os
from src.data import FloorPlan

if __name__ == '__main__':
    data_directory = os.path.join(os.getcwd(), 'data')
    floorplan = FloorPlan(1, 'F3', data_directory)
    floorplan.plot_wifi_heatmap(False, 'dBm')
    floorplan.show()