import os
import matplotlib.pyplot as plt
import math
import pandas as pd


def euler_distance(x, y):
    return math.sqrt((x[0]-y[0]) ** 2 + (x[1] - y[1]) ** 2)

def convertList(list):
    string = str(list)
    return string.replace(' ', '')

def set_threshold():
    global threshold_min, threshold_max, threshold_step
    threshold_min = 0.01
    threshold_max = 0.02
    threshold_step = 0.002

train_1000 = pd.read_csv('data/train_1000.csv')
consecutive_dis = []


for n in range(1000):
    gps_points = eval(train_1000['POLYLINE'][n])
    for i in range(len(gps_points)-1):
        pre = gps_points[i]
        post = gps_points[i+1]
        cur_dis = euler_distance(pre, post)
        consecutive_dis.append(cur_dis)



plt.hist(x = consecutive_dis, 
    bins = 20,
    color = 'steelblue',
    edgecolor = 'black',
    range=(0,0.015)
    )

plt.savefig('output/task_6/distribution', dpi=240)


set_threshold()


def calculateThreshold():
    global last_point, threshold
    next_point = gps_points[i]
    if euler_distance(last_point, next_point) < threshold:
        improved_ps.append(next_point)
        last_point = next_point
        threshold = threshold_min
    else:
        threshold += threshold_step
        threshold = min(threshold, threshold_max)


for n in range(1000):
    gps_points = eval(train_1000['POLYLINE'][n])
    assert isinstance(train_1000['POLYLINE'][n], str)

    if len(gps_points) < 2: 
        improved_ps = gps_points
    else:
        improved_ps = gps_points[:1]
        last_point = gps_points[0]
        threshold = threshold_min
        for i in range(1, len(gps_points)):
            calculateThreshold()
        train_1000['POLYLINE'][n] = convertList(improved_ps)

train_1000.to_csv('data/improved_train_1000.csv',index=0)