import os
from src.data import FloorPlan, RootDirectory

if __name__ == '__main__':
    data_directory = os.path.join(os.getcwd(), 'data')
    output_directory = os.path.join(os.getcwd(), 'output')
    rd = RootDirectory()
    available_data = rd._AVAILABLE_DATA
    #available_data = {'site1':['B1']}
    output = []
    for k, v in available_data.items():
        for floor in v:
            site_num = int(k[-1])
            floorplan = FloorPlan(site_num, floor, data_directory)
            
            data = floorplan.describe_waypoints(False)
            print(data)
            data = [k+'_'+floor] + list(data)
            output.append(','.join(data))
    
    path = os.path.join(output_directory, '_'.join([k, floor])+'.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(output))

    output = []
    for k, v in available_data.items():
        for floor in v:
            site_num = int(k[-1])
            floorplan = FloorPlan(site_num, floor, data_directory)
            
            data = floorplan.describe_waypoints(True)
            print(data)
            data = [k+'_'+floor] + list(data)
            output.append(','.join(data))
    
    path = os.path.join(output_directory, '_'.join([k, floor, 'augmented'])+'.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(output))

            # floorplan.plot_trajectory('lines + markers', augment_data = False)
            # floorplan.show(display=False)
            # path = os.path.join(output_directory, '_'.join([k, floor, 'raw'])+'.jpeg')
            # floorplan.figure.write_image(path)

            # floorplan = FloorPlan(site_num, floor, data_directory)
            # floorplan.plot_trajectory('lines + markers', augment_data = True)
            # floorplan.show(display=False)
            # path = os.path.join(output_directory, '_'.join([k, floor, 'augmented'])+'.jpeg')
            # floorplan.figure.write_image(path)