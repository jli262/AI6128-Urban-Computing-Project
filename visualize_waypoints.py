import os
from src.data import FloorPlan

if __name__ == '__main__':
    data_directory = os.path.join(os.getcwd(), 'data')

    floorplan = FloorPlan(1, 'B1', data_directory)
    floorplan.plot_trajectory('lines + markers')
    floorplan.show()
    floorplan.figure.write_image(floorplan.directory + '/fig1.jpeg')
