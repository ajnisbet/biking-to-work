import json
import math
import os

from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


OSRM_URL = 'http://192.168.99.100:5000'
OSRM_BATCH_SIZE = 500  # Max number of coordinates in a single request.


def main():
    '''
    Produce plots of travel time contours around a given point.
    '''

    # Parameters
    point_spacing = 200  # How far apart points shoud be, in metres.
    grid_size = 200  # Width and height of the grid, in number of points.
    work_lat = 37.388071
    work_lon = -122.055957
    cluster_threshold = point_spacing * 0.75  # Max closeness of 2 points.
    smoothing_factor = 15  # Gaussian blur parameter.
    resample_factor = 40  # Resolution of interpolated gird, relative to point spacing.
    contour_levels = [15, 30, 45]  # In minutes.

    # Build grid.
    #
    # The latlon positions are found for each edge of the grid, then points
    # are evenly spaced beween the extremes.
    grid_radius = grid_size * point_spacing / 2
    north_lat, _ = offset_lat_lon(work_lat, work_lon, grid_radius, 0)
    _, east_lon = offset_lat_lon(work_lat, work_lon, grid_radius, 90)
    south_lat, _ = offset_lat_lon(work_lat, work_lon, grid_radius, 180)
    _, west_lon = offset_lat_lon(work_lat, work_lon, grid_radius, 270)
    lat_values = np.linspace(south_lat, north_lat, grid_size)
    lon_values = np.linspace(west_lon, east_lon, grid_size)
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values)

    # Grid distortion.
    #
    # Taking evenly spaced points in the latlon space results in warping
    # compared to the xy space. The distortion is the spacing between points
    # at the top of the grid compared to the spacing at the bottom.
    north_width = lat_lon_distance(north_lat, west_lon, north_lat, east_lon)
    south_width = lat_lon_distance(south_lat, west_lon, south_lat, east_lon)
    distortion = (south_width - north_width) / north_width
    print('Max grid distortion: {:.2%}'.format(distortion))

    # Flatten grid.
    #
    # Once points start being moved, it doesn't make sense to keep them in a
    # 2D array.
    lats = []
    lons = []
    for i, _ in np.ndenumerate(lat_grid):
        lats.append(lat_grid[i])
        lons.append(lon_grid[i])

    # Snap to nearest valid point.
    snapped_lats = []
    snapped_lons = []
    for lat, lon in zip(lats, lons):
        coords = '{:.6f},{:.6f}'.format(lon, lat)
        url = '{}/nearest/v1/cycling/{}'.format(OSRM_URL, coords)
        result = get_json(url)
        snapped_lats.append(result['waypoints'][0]['location'][1])
        snapped_lons.append(result['waypoints'][0]['location'][0])

    lats = np.array(snapped_lats)
    lons = np.array(snapped_lons)

    # Clip to circle.
    distances = lat_lon_distance(work_lat, work_lon, lats, lons)
    lats = lats[distances < grid_radius]
    lons = lons[distances < grid_radius]

    # Pruning.
    #
    # Any point that's within a threshold of another point is removed. Once a
    # point is marked as removed, it isn't used for comparison. It's a lazy
    # rule, but it beats using a proper clustering algorithm like k-means
    # because you know that the resulting clusters are valid bikable points.
    remove_point = np.full(len(lats), False, dtype=bool)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if remove_point[i]:
            continue
        distances = lat_lon_distance(lat, lon, lats[i+1:], lons[i+1:])
        remove_point[i+1:] = remove_point[i+1:] | (distances < cluster_threshold)

    lats = lats[~remove_point]
    lons = lons[~remove_point]

    # Travel times.
    durations = []
    for i in range(0, len(lats), OSRM_BATCH_SIZE):
        batch_lats = [work_lat] + list(lats[i:i + OSRM_BATCH_SIZE])
        batch_lons = [work_lon] + list(lons[i:i + OSRM_BATCH_SIZE])
        coords = ';'.join('{:.6f},{:.6f}'.format(lon, lat) for lon, lat in zip(batch_lons, batch_lats))
        url = '{}/table/v1/cycling/{}?sources=0'.format(OSRM_URL, coords)
        result = get_json(url)
        durations += result['durations'][0][1:]
    durations = np.array(durations)

    # Resampling.
    #
    # Although matplotlib can do contours for non-uniform grids, it's
    # difficult to smooth something like that. Instead, the distance values
    # are interpolated back onto an even grid.
    x_values = np.linspace(min(lons), max(lons), grid_size * resample_factor)
    y_values = np.linspace(min(lats), max(lats), grid_size * resample_factor)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.array([lons, lats]).T
    grid_points = np.array([x_grid, y_grid]).T
    z_grid = interpolate.griddata(points, durations, grid_points, method='linear').T

    # Contours.
    #
    # The longest contour for each level is printed.
    fig, ax = plt.subplots()
    levels = np.array(contour_levels) * 60
    z_grid_smooth = ndimage.filters.gaussian_filter(z_grid, smoothing_factor)
    contours = ax.contour(x_grid, y_grid, z_grid_smooth, levels=levels, antialiased=True)
    for i, collection in enumerate(contours.collections):
        paths = collection.get_paths()
        paths.sort(key=lambda x: len(x.vertices))
        path = paths[-1]
        coords = path.vertices.tolist()
        print coords    




def get_json(url):
    '''
    GET a url, and return the parsed json.
    '''
    return json.loads(requests.get(url).text)


def radius_of_earth(lat):
    '''
    Find the radius of the Earth at a given latitude. Vectorised.

    Formula and data from https://en.wikipedia.org/wiki/Earth_radius
    '''
    r_1 = 6378137.0  # Earth equatorial radius
    r_2 = 6356752.3  # Earth polar radius

    lat = np.radians(lat)
    part_1 = r_1**4 * np.cos(lat)**2 + r_2**4 * np.sin(lat)**2
    part_2 = r_1**2 * np.cos(lat)**2 + r_2**2 * np.sin(lat)**2

    return np.sqrt(part_1 / part_2)


def offset_lat_lon(lat, lon, dist, bearing):
    '''
    Shift a latlon point by a Cartesian vector.

    From github.com/kellydunn/golang-geo/blob/master/point.go
    '''
    dist_ratio = dist / radius_of_earth(lat)
    bearing = math.radians(bearing)
    lat_1 = math.radians(lat)
    lon_1 = math.radians(lon)

    lat_2_part_1 = math.sin(lat_1) * math.cos(dist_ratio)
    lat_2_part_2 = math.cos(lat_1) * math.sin(dist_ratio) * math.cos(bearing)
    lat_2 = math.asin(lat_2_part_1 + lat_2_part_2)

    lon_2_part_1 = math.sin(bearing) * math.sin(dist_ratio) * math.cos(lat_1)
    lon_2_part_2 = math.cos(dist_ratio) - (math.sin(lat_1) * math.sin(lat_2))
    lon_2 = lon_1 + math.atan2(lon_2_part_1, lon_2_part_2)

    lat_2 = math.degrees(lat_2)
    lon_2 = math.degrees(lon_2)
    return lat_2, lon_2



def lat_lon_distance(from_lat, from_lon, to_lat, to_lon):
    '''
    Find the distance between two latlon points.

    Uses the Haversine forumula.

    to_lat and to_lon can be vectors of points.
    '''
    mean_lat = (from_lat + to_lat) / 2
    radius = radius_of_earth(mean_lat)

    from_lat = math.radians(from_lat)
    from_lon = math.radians(from_lon)
    to_lat = np.radians(to_lat)
    to_lon = np.radians(to_lon)
    d_lat = to_lat - from_lat
    d_lon = to_lon - from_lon

    a = np.sin(d_lat / 2)**2 + math.cos(from_lat) * np.cos(to_lat) * np.sin(d_lon  / 2)**2
    d = radius * 2 * np.arcsin(np.sqrt(a))
    return d


if __name__ == '__main__':
    main()