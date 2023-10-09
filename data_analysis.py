#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project, Task 1: Feature Engineering - Data Visualization
# 
# Before you start, make sure that you are familiar with the basic usage of Jupyter Notebook. 
# 
# If not, please finish the Jupyter Notebook primer first. Additionally, visit the Azure Notebook library [Cloud Computing Course](https://notebooks.azure.com/CloudComputingCourse/projects/cloud-computing-course) and read the tutorials with **worked examples** and practice on Linux, Bash and Pandas.
# 
# In this task, you are visualizing spatial and temporal data which would influence cab fare prices in New York City. You need to implement the following methods based on your observations:
# ```
# q1()
# q2()
# q3()
# q4()
# ```
# 
# Please do not change any utility method. More cells may be added to the notebook. If you don't want to include the cell in the converted script, please tag the cell with `excluded_from_script`. You can display the tags for each cell as such: `View > Cell Toolbar > Tags`.
# 
# Execute `./runner.sh` in the console to check the result. Please make sure that the virtualenv is activated when executing `runner.sh`.
# 

# In[1]:


# Import libraries that q1 - q4 depend on.
# Please DO NOT change this cell. 
# The cell will be included in the converted Python script.
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
import scipy.signal
import sys
import argparse
import os


# In[3]:


os.environ['MAPBOX_ACCESS_TOKEN'] = "pk.eyJ1IjoieXdhbjk0NiIsImEiOiJjbGJlbXJicjUwY2p3M29xcmJ5cW10aWo5In0.Cb_I-uC0-7TTwhl4BIwD4g"


# # Q1: Spatial Data Visualization
# >In q1, you will explore the geographical distribution of the data.
# 
# >You should carry out the data visualization and explore the dataset using `cc_nyc_fare_train_small.csv`, the same dataset will be used in the following feature engineering task. However, please use `NA_boundary_box.csv` when submitting q1.
# 
# >Steps:
# >1. Find the proper inputs to feed into the `visualize_map` method and visualize the spatial data.
# >2. Explore the data points on the map. Does every point make sense? Should some data be in the Atlantic Ocean? 
# >3. Implement a data filter to exclude rows with pickup location outside the region of the United States.
# 
# >Hint: 
# 
# >You may want to figure out a boundary of latitude and longitude for the region of the United States excluding the bordering countries. A good place to find bounding box is: http://boundingbox.klokantech.com/. Please explore the usage of this bounding box tool and find the required bounding box. You may drag and drop the box to include the region you want.

# In[7]:


def q1():
    """
    ML Objective: When exploring raw datasets you will often come across data points which do not fit the business 
    case and are called outliers. In this case, the outliers might denote data points outside of the specific area
    since our goal is to develop a model to predict fares in NYC. 
    
    You might want to exclude such data points to make your model perform better in the Feature Engineering Task.
    
    TODO: Exclude rows with pickup location outside the region of the United States excluding the bordering countries.
    
    output format:
    <row number>, <pickup_longitude>, <pickup_latitude>
    <row number>, <pickup_longitude>, <pickup_latitude>
    <row number>, <pickup_longitude>, <pickup_latitude>
    ...
    """
    
    df = pd.read_csv('data/NA_boundary_box.csv').loc[:,['pickup_latitude', 'pickup_longitude']]
    # -74.172151,40.583016,-73.805795,40.868175
    # TODO: Implement a data filter to exclude the data outside the region of the United States
    #       (replace "None" with your implementation)
#     raise NotImplementedError("To be implemented")
    res = df.loc[(df.pickup_longitude.between(-74.172151, -73.805795)) & (df.pickup_latitude.between(40.583016, 40.868175))]
    
    # print the result to standard output in the CSV format
    res.to_csv(sys.stdout, encoding='utf-8', header=None)


# In[29]:


# Utility methods, please do not change.

def haversine_distance(origin, destination):
    """
    Formula to calculate the spherical distance between 2 coordinates, with each specified as a (lat, lng) tuple

    :param origin: (lat, lng)
    :type origin: tuple
    :param destination: (lat, lng)
    :type destination: tuple
    :return: haversine distance
    :rtype: float
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d
  

def draw_heatmap(data, center, zoom):
    """
    Method to draw a heat map. You should use this method to identify the most popular pickup location in the southeast of NYC.

    :param geodata: name of GeoJSON file or object or JSON join-data weight_property
    :type geodata: string
    :param center: map center point
    :type center: tuple
    :param zoom: starting zoom level for map
    :type zoom: float
    """
    # set features for the heatmap
    heatmap_color_stops = create_color_stops([0.01,0.25,0.5,0.75,1], colors='RdPu')
    heatmap_radius_stops = [[10,1],[20,2]] #increase radius with zoom

    # create a heatmap
    viz = HeatmapViz(data,
                     access_token=os.environ['MAPBOX_ACCESS_TOKEN'],
                     color_stops=heatmap_color_stops,
                     radius_stops=heatmap_radius_stops,
                     height='500px',
                     opacity=0.9,
                     center=center,
                     zoom=zoom)
    print("drawing map...")
    viz.show()


# In[34]:


def q2():
    """
    ML Objective: When exploring raw datasets, you will come often across a small set of data points which might 
    exhibit a unique or different behavior as compared to the rest of the data points. 
    
    In this case, the fare between two hotspots in NYC might be much higher irrespective of the distance between them. 
    You might want to identify such data points to make your model perform better during the Feature Engineering Task.
    
    TODO: calculate the distance between MSG and the most popular pickup location in the southeast of NYC.
    
    output format:
    <distance>
    """
   
    MSG_coor = (40.750298, -73.993324) # lat, lng
    
    # TODO: replace "None" with your implementation
#     raise NotImplementedError("To be implemented")
    hot_spot_coor = (40.6444, -73.7812)
    res = haversine_distance(MSG_coor, hot_spot_coor)
    
    print(round(res, 2))


# In[14]:


def q3():
    """
    ML Objective: As described above, time based features are crucial for better performance of an ML model since 
    the input data points often change with respect to time.  
    
    In this case, the traffic conditions might be higher during office hours or during weekends which may result 
    in higher fares. You might want to develop such time-based features to make your model perform better during the 
    Feature Engineering Task.
    
    TODO: You need to implement the method to extract year, month, hour and weekday from the pickup_datetime feature
    
    output format:
    <row number>, <pickup_datetime>, <fare_amount>, <year>, <month>, <hour>, <weekday>
    """
    # read the CSV file into a DataFrame
    df = pd.read_csv('data/cc_nyc_fare_train_tiny.csv', parse_dates=['pickup_datetime'])
    time_features = df.loc[:, ['pickup_datetime', 'fare_amount']]

    # TODO: extract time-related features from the `pickup_datetime` column.
    #       (replace "None" with your implementation)
#     raise NotImplementedError("To be implemented")
    time_features['year'] = pd.DatetimeIndex(df['pickup_datetime']).year
    time_features['month'] = pd.DatetimeIndex(df['pickup_datetime']).month
    time_features['hour'] = pd.DatetimeIndex(df['pickup_datetime']).hour
    time_features['weekday'] = pd.DatetimeIndex(df['pickup_datetime']).weekday
    
    # print the result to standard output in the CSV format
    time_features.to_csv(sys.stdout, encoding='utf-8', header=None)


# In[68]:


def q4():
    """
    ML Objective: While relying on time based features might be beneficial, it is a good practice to remove the 
    abnormalities in the data. 
    
    In this case, the time of the day might not be an explicable factor for the resulting fare. When developing 
    time-based features you might want to exclude a few abnormal data points which might lead to overfitting.
    
    Fix the abnormal distribution in `fare_amount` by removing 0.1% of raw data.
    
    output format:
    <row number>, <pickup_datetime>, <fare_amount>
    <row number>, <pickup_datetime>, <fare_amount>
    <row number>, <pickup_datetime>, <fare_amount>
    ...
    """
    # read the CSV file into a DataFrame
    df = pd.read_csv('data/cc_nyc_fare_train_tiny.csv', parse_dates=['pickup_datetime']).loc[:, ['pickup_datetime', 'fare_amount']]

    # TODO: replace "None" with the 99.9% quantile
#     raise NotImplementedError("To be implemented")
    df = df[df['fare_amount'] < df['fare_amount'].quantile(.999, interpolation='lower')]

    # print the result to standard output in the CSV format
    df.to_csv(sys.stdout, encoding='utf-8', header=None)


# In[ ]:


def main():
    parser = argparse.ArgumentParser(
        description="Project Machine Learning on Cloud")
    parser.add_argument("-r",
                        metavar='<question_id>',
                        required=False)
    args = parser.parse_args()
    question = args.r

    if question == "q1":
        q1()
    elif question == "q2":
        q2()
    elif question == "q3":
        q3()
    elif question == "q4":
        q4()
    else:
        print("Invalid question")

if __name__ == "__main__":
    main()

