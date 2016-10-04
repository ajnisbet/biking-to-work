import hashlib
import json
import math
import os
import pickle
import tempfile

import requests
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt


OSRM_URL = 'http://192.168.99.100:5000'
json.loads(requests.get(url).text)


def get_json(url):
    return json.loads(requests.get(url).text)


def radius_of_earth(lat):
    '''
    Find the radius of the Earth at a given latitude.

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
    # lon_2 = math.fmod(lon_2 + 3 * math.pi, 2 * math.pi) - math.pi

    lat_2 = math.degrees(lat_2)
    lon_2 = math.degrees(lon_2)
    return lat_2, lon_2



def lat_lon_distance(from_lat, from_lon, to_lat, to_lon):
    '''
    Find the distance between two latlon points.
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


def main():
    '''
    Run the code, saving the results files.
    '''

    # Parameters
    point_spacing = 200  # Meters
    grid_size = 200  # Number of points
    work_lat = 37.388071
    work_lon = -122.055957
    osrm_batch_size = 500
    rough_smoothing_factor = 15
    smooth_smoothing_factor = 50
    resample_factor = 40
    contour_levels = [15, 30, 45]
    duration_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    cluster_spacing = point_spacing * 0.8

    # Build grid
    grid_radius = grid_size * point_spacing // 2
    north_lat, _ = offset_lat_lon(work_lat, work_lon, grid_radius, 0)
    _, east_lon = offset_lat_lon(work_lat, work_lon, grid_radius, 90)
    south_lat, _ = offset_lat_lon(work_lat, work_lon, grid_radius, 180)
    _, west_lon = offset_lat_lon(work_lat, work_lon, grid_radius, 270)
    lat_values = np.linspace(south_lat, north_lat, grid_size)
    lon_values = np.linspace(west_lon, east_lon, grid_size)
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values)

    # Estimate grid distortion
    north_width = lat_lon_distance(north_lat, west_lon, north_lat, east_lon)
    south_width = lat_lon_distance(south_lat, west_lon, south_lat, east_lon)
    distortion = (south_width - north_width) / north_width
    print('Max grid distortion: {:.2%}'.format(distortion))

    # Flatten grid
    lats = []
    lons = []
    for i, _ in np.ndenumerate(lat_grid):
        lats.append(lat_grid[i])
        lons.append(lon_grid[i])

    # Snap to nearest valid lat/lon for biking
    snapped_lats = []
    snapped_lons = []
    for lat, lon in zip(lats, lons):
        coords = '{:.6f},{:.6f}'.format(lon, lat)
        url = '{}/nearest/v1/cycling/{}'.format(OSRM_URL, coords)
        result = cached_get(url)
        snapped_lats.append(result['waypoints'][0]['location'][1])
        snapped_lons.append(result['waypoints'][0]['location'][0])

    # Clip to circle
    lats = np.array(snapped_lats)
    lons = np.array(snapped_lons)
    distances = lat_lon_distance(work_lat, work_lon, lats, lons)
    lats = lats[distances < grid_radius]
    lons = lons[distances < grid_radius]

    # Declustering
    remove_point = np.full(len(lats), False, dtype=bool)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if remove_point[i]:
            continue
        distances = lat_lon_distance(lat, lon, lats[i+1:], lons[i+1:])
        remove_point[i+1:] = remove_point[i+1:] | (distances < cluster_spacing)

    lats = lats[~remove_point]
    lons = lons[~remove_point]

    # Durations
    durations = []
    for i in range(0, len(lats), osrm_batch_size):
        batch_lats = [work_lat] + list(lats[i:i + osrm_batch_size])
        batch_lons = [work_lon] + list(lons[i:i + osrm_batch_size])
        coords = ';'.join('{:.6f},{:.6f}'.format(lon, lat) for lon, lat in zip(batch_lons, batch_lats))
        url = '{}/table/v1/cycling/{}?sources=0'.format(OSRM_URL, coords)
        result = cached_get(url)
        durations += result['durations'][0][1:]
    durations = np.array(durations)

    # Resample the duration to a regular grid
    x_values = np.linspace(min(lons), max(lons), grid_size * resample_factor)
    y_values = np.linspace(min(lats), max(lats), grid_size * resample_factor)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Make the actual contours    
    fig, ax = plt.subplots()
    points = np.array([lons, lats]).T
    grid_points = np.array([x_grid, y_grid]).T
    z_grid = interpolate.griddata(points, durations, grid_points, method='linear').T
    levels = np.array(contour_levels) * 60
    z_grid_smooth = ndimage.filters.gaussian_filter(z_grid, smooth_smoothing_factor)
    contours = ax.contour(x_grid, y_grid, z_grid_smooth, levels=levels, antialiased=True)
    

if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)
    main()