import os
from src.data import FloorPlan, RootDirectory

if __name__ == '__main__':
    data_directory = os.path.join(os.getcwd(), 'data')
    output_directory = os.path.join(os.getcwd(), 'output')
    rd = RootDirectory()
    available_data = rd._AVAILABLE_DATA
    for k, v in available_data.items():
        for floor in v:
            site_num = int(k[-1])
            floorplan = FloorPlan(site_num, floor, data_directory)
            floorplan.plot_trajectory('lines + markers')
            floorplan.show()
            path = os.path.join(output_directory, '_'.join([k, floor])+'.jpeg')
            floorplan.figure.write_image(path)
