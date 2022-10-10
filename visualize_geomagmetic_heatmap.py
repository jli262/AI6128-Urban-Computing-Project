import os
from src.data import FloorPlan, RootDirectory

if __name__ == '__main__':
    data_directory = os.path.join(os.getcwd(), 'data')  # Path to input data
    output_directory = os.path.join(os.getcwd(), 'output/magnetic')  # Path to output data
    rd = RootDirectory()  # Initialization Class - RootDirectory
    available_data = rd._AVAILABLE_DATA  # Site、Floor
    for k, v in available_data.items():  # Iterate over the K-Values k:site1,site2 value list:['B1', 'F1', 'F2', 'F3']
        for floor in v:  # Iterate ['B1', 'F1', 'F2', 'F3', 'F4']
            site_num = int(k[-1])  # Take the last character: 1,2
            floorplan = FloorPlan(site_num, floor, data_directory)  # floorplan Object
            floorplan.plot_magnetic('lines + markers')
            floorplan.show()  # Display via webpage
            path = os.path.join(output_directory, '_'.join([k, floor]) + '.jpeg')
            floorplan.figure.write_image(path)  # Save Images
