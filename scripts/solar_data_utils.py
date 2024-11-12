# Functions for my work

import os
import glob
seed_val = 7
import os
os.environ['PYTHONHASHSEED'] = str(seed_val)
import random
random.seed(seed_val)
import numpy as np
np.random.seed(seed_val)
from scipy.optimize import fsolve as fsolver
import pandas as pd
import sunpy
from astropy import units as u
from sunpy.net import Fido, attrs as a
import requests
from bs4 import BeautifulSoup
from spacepy import pycdf
from scipy.ndimage import gaussian_filter
from astropy.visualization import ImageNormalize, PercentileInterval
from sunpy import timeseries as ts
import matplotlib.dates as mdates
from PIL import Image
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolors

data_dir = '/home/mnedal/data'











def newkirk(r):
    """
    Newkirk electron-density model with fundamental emission.
    `fold` is a multiplicative factor to change the density scaling.
    """
    fold = 1
    return fold*4.2e4*10.**(4.32/r)


def omega_pe_r(ne_r, r):
    ''' 
    Plasma frequency density relationship.
    Works only for the fundamental emission.
    ''' 
    return 8.93e3*(ne_r(r))**(0.5)*2*np.pi


def freq_to_R(f_pe, ne_r=newkirk):
    """
    Starting height for a wave frequency.
    """
    func = lambda R: f_pe - (omega_pe_r(ne_r, R))/2/np.pi
    R_solution = fsolver(func, 1.5) # solve the R
    return R_solution # in solrad unit


def get_colors(n):
    """
    Generate a list of (n) hex gradiant colors.
    """
    cmap   = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, n)]
    hex_colors = [mcolors.to_hex(c) for c in colors]
    hex_colors[0]  = '#0000ff' # set first color to blue
    hex_colors[-1] = '#ff0000' # set last color to red
    return hex_colors


def draw_halfSun():
    theta, phi = np.mgrid[0:np.pi:100j, 0:np.pi:100j]
    x = np.sin(theta)*np.cos(phi) 
    y = np.sin(theta)*np.sin(phi) 
    z = np.cos(theta)
    # applying rotation matrix to rotate the half of the Sun
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(np.deg2rad(0)), -np.sin(np.deg2rad(0))],
                   [0, np.sin(np.deg2rad(0)), np.cos(np.deg2rad(0))]])
    x, y, z = np.einsum('ij,jkl->ikl', Rx, np.array([x, y, z]))
    Rz = np.array([[np.cos(np.deg2rad(-90)), -np.sin(np.deg2rad(-90)), 0],
                   [np.sin(np.deg2rad(-90)), np.cos(np.deg2rad(-90)), 0],
                   [0, 0, 1]])
    x, y, z = np.einsum('ij,jkl->ikl', Rz, np.array([x, y, z]))
    return x, y, z



def draw_fullSun():
    theta, phi = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
    x = np.sin(theta)*np.cos(phi) 
    y = np.sin(theta)*np.sin(phi) 
    z = np.cos(theta)
    return x, y, z



def split_datetime(start=None, end=None):
    START_DATE, START_TIME = start.split('T')
    END_DATE, END_TIME     = end.split('T')

    START_YEAR, START_MONTH, START_DAY = START_DATE.split('-')
    END_YEAR, END_MONTH, END_DAY       = END_DATE.split('-')

    START_HOUR, START_MINUTE, START_SECOND = START_TIME.split(':')
    END_HOUR, END_MINUTE, END_SECOND       = END_TIME.split(':')

    datetime_dict = {
        'start_year': START_YEAR,
        'start_month': START_MONTH,
        'start_day': START_DAY,
        'start_hour': START_HOUR,
        'start_minute': START_MINUTE,
        'start_second': START_SECOND,
        
        'end_year': END_YEAR,
        'end_month': END_MONTH,
        'end_day': END_DAY,
        'end_hour': END_HOUR,
        'end_minute': END_MINUTE,
        'end_second': END_SECOND
    }
    return datetime_dict




def plot_line(angle_deg=None, length=None, map_obj=None):
    """
    Plot a straight line at an angle in degrees from the solar West.
    """
    angle_rad = np.deg2rad(angle_deg)
    
    # Define the length of the line (in arcseconds)
    line_length = length * u.arcsec
    
    # Define the center point of the line (e.g., the center of the Sun)
    center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=map_obj.coordinate_frame)
    
    # Calculate the start and end points of the line
    start_point = SkyCoord(center.Tx, center.Ty, frame=map_obj.coordinate_frame)
    end_point   = SkyCoord(center.Tx + line_length * np.cos(angle_rad), 
                           center.Ty + line_length * np.sin(angle_rad),
                           frame=map_obj.coordinate_frame)
    
    line = SkyCoord([start_point, end_point])
    return line



def generate_centered_list(center, difference, num_elements):
    """
    Generate a list of numbers centered around a given number with a specified difference
    between consecutive numbers.

    Parameters:
    center (int): The central number around which the list is generated.
    difference (int): The difference between consecutive numbers in the list.
    num_elements (int): The number of elements before and after the central number.

    Returns:
    list: A list of numbers centered around the specified central number.
    """
    return [center + difference * i for i in range(-num_elements, num_elements + 1)]



def remove_redundant_maps(maps):
    """
    Remove redundant SunPy maps, keeping only one map per unique timestamp.

    Parameters:
    maps (list): List of SunPy Map objects. Each map is expected to have a 'date-obs' 
                 key in its metadata that provides the observation timestamp.

    Returns:
    list: A list of unique SunPy Map objects, one per unique timestamp.
    
    Example:
    >>> unique_maps = remove_redundant_maps(list_of_sunpy_maps)
    """
    unique_maps = {}
    for m in maps:
        timestamp = m.latex_name
        if timestamp not in unique_maps:
            unique_maps[timestamp] = m
    return list(unique_maps.values())



def apply_runratio(maps):
    """
    Apply running-ratio image technique on EUV images.
    See: https://iopscience.iop.org/article/10.1088/0004-637X/750/2/134/pdf
        Inputs: list of EUV sunpy maps.
        Output: sequence of run-ratio sunpy maps.
    """
    runratio = [m / prev_m.quantity for m, prev_m in zip(maps[1:], maps[:-1])]
    m_seq_runratio = sunpy.map.Map(runratio, sequence=True)
    
    for m in m_seq_runratio:
        m.data[np.isnan(m.data)] = 1
        m.plot_settings['norm'] = colors.Normalize(vmin=0, vmax=2)
        m.plot_settings['cmap'] = 'Greys_r'
    
    return m_seq_runratio



def enhance_contrast(image, vmin, vmax):
    """
    Enhance contrast by clipping and normalization.
    """
    image_clipped = np.clip(image, vmin, vmax)
    image_normalized = (image_clipped - vmin) / (vmax - vmin)
    return image_normalized



def calculate_percentiles(image, lower_percentile=3, upper_percentile=97):
    """
    Calculate vmin and vmax based on the 1st and 99th percentiles.
    """
    vmin = np.percentile(image, lower_percentile)
    vmax = np.percentile(image, upper_percentile)
    return vmin, vmax



def round_obstime(time=None):
    """
    Round the observation time to put it in the image title.
    Input : str, time (HH:MM:SS.sss)
    Output: str, datetime (YYYY-mm-DD HH:MM:SS)
    """
    from datetime import datetime, timedelta

    original_time_str = time

    # Convert the original time string to a datetime object
    original_time = datetime.strptime(original_time_str, '%H:%M:%S.%f')

    # Round the time to the nearest second
    rounded_time = original_time + timedelta(seconds=round(original_time.microsecond / 1e6))

    # Format the rounded time as 'HH:MM:SS'
    rounded_time_str = rounded_time.strftime('%H:%M:%S')
    
    return rounded_time_str



def load_lasco(data_dir=None, start=None, end=None, detector='C2'):
    """
    Load SOHO/LASCO C2 or C3 images as sunpy maps.
    """
    dt_dict = split_datetime(start=start, end=end)
    data = sorted(glob.glob(f"{data_dir}/LASCO_{detector}/LASCO_{detector}_{dt_dict['start_year']}{dt_dict['start_month']}{dt_dict['start_day']}*.jp2"))
    
    start_file_to_find = f"{data_dir}/LASCO_{detector}/LASCO_{detector}_{dt_dict['end_year']}{dt_dict['start_month']}{dt_dict['start_day']}T{dt_dict['start_hour']}{dt_dict['start_minute']}.jp2"
    end_file_to_find = f"{data_dir}/LASCO_{detector}/LASCO_{detector}_{dt_dict['end_year']}{dt_dict['end_month']}{dt_dict['end_day']}T{dt_dict['end_hour']}{dt_dict['end_minute']}.jp2"
    
    idx1 = data.index(start_file_to_find)
    idx2 = data.index(end_file_to_find)
    chosen_files = data[idx1:idx2]
    
    map_objects = []
    for i, file in enumerate(chosen_files):
        m = sunpy.map.Map(file)
        m.meta['bunit'] = 'ct' # a workaround for C2 and C3 jp2 images
        m.plot_settings['norm'] = ImageNormalize(vmin=0, vmax=250)
        map_objects.append(m)
        print(f'LASCO {detector} image {i} is done')
    return map_objects



def generate_number_list_float(center, offset, count):
    """
    Generate a list of numbers around a given center number.

    Parameters:
    center (int or float): The central number of the list.
    offset (int or float): The increment by which numbers are spaced from the center.
    count (int): The number of numbers to include on each side of the center number.

    Returns:
    list: A list of numbers around the given center.

    Example usage:
        center_number = 0
        offset_value = 2
        count_around = 3
        number_list = generate_number_list(center_number, offset_value, count_around)
        print(number_list)
        [-6, -4, -2, 0, 2, 4, 6]
    """
    return [center + offset * i for i in range(-count, count + 1)]



def generate_number_list_time(center, offset, count):
    """
    Generate a list of time numbers around a given center date number.

    Parameters:
    center (int or float): The central date number of the list.
    offset (int or float): The increment by which numbers are spaced from the center, in minutes.
    count (int): The number of numbers to include on each side of the center number.

    Returns:
    list: A list of date numbers around the given center.

    Example usage:
        center_number = 19857.763069 # equivalent to 2024-05-14 18:18:49.161600+00:00
        offset_value = 2 # in minutes
        count_around = 2
        number_list = generate_number_list_time(center_number, offset_value, count_around)
        print(number_list)
    """
    # Convert center date number to datetime object
    center_date = mdates.num2date(center)

    # Generate list of date numbers
    list_dates = []
    for i in range(-count, count + 1):
        # Calculate the new date by adding the offset in minutes
        new_date = center_date + timedelta(minutes=offset * i)
        # Convert the datetime object back to a date number
        new_date_number = mdates.date2num(new_date)
        list_dates.append(new_date_number)
    
    return list_dates



def onclick(event):
    """
    This function is called when the mouse is clicked on the figure.
    It adds the x and y coordinates of the click to the coords list.
    """
    global current_trial, text_handle

    if event.button == 1:  # Left mouse button
        xx, yy = event.xdata, event.ydata  # Get the central x and y coordinates
        ax.plot(xx, yy, 'ro', markersize=7)
        plt.draw()

        # Store the coordinates in the current trial's list
        feature_coords_slit[f'trial_{current_trial}'].append((xx, yy))
        
        # Update the text on the plot
        if text_handle:
            text_handle.remove()  # Remove the previous text
        text_handle = ax.text(0.05, 0.95, f"Trial {current_trial + 1}: Captured ({xx:.2f}, {yy:.2f})", 
                              transform=ax.transAxes, fontsize=12, verticalalignment='top', color='pink')
        plt.draw()

    elif event.button == 3:  # Right mouse button
        # Move to the next trial
        current_trial += 1
        if current_trial >= num_repeats:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)  # Close the figure window
        else:
            if text_handle:
                text_handle.remove()
            text_handle = ax.text(0.05, 0.95, f"Moving to trial {current_trial + 1}", 
                                  transform=ax.transAxes, fontsize=12, verticalalignment='top', color='cyan')
            plt.draw()


def compute_standard_error(coords_dict):
    # Extract all trials
    trials = list(coords_dict.values())
    num_points = len(trials[0])  # Number of points in each trial

    # Initialize lists to store standard errors
    mean_values = []
    standard_errors = []

    # Loop over each point position
    for point_idx in range(num_points):
        # Extract x and y values for this point across all trials
        x_values = [trials[trial_idx][point_idx][0] for trial_idx in range(len(trials))]
        y_values = [trials[trial_idx][point_idx][1] for trial_idx in range(len(trials))]

        # Convert to numpy arrays
        x_values = np.array(x_values)
        y_values = np.array(y_values)

        # Compute mean and standard error for this point
        mean_x = np.mean(x_values)
        mean_y = np.mean(y_values)
        se_x = np.std(x_values, ddof=1) / np.sqrt(len(x_values))
        se_y = np.std(y_values, ddof=1) / np.sqrt(len(y_values))

        # Store results
        mean_values.append((mean_x, mean_y))
        standard_errors.append((se_x, se_y))

    return mean_values, standard_errors







def png_to_gif(path=None):
    # set the path to the folder containing the PNG images
    # get a list of all PNG files in the folder
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    files.sort(key=lambda x: (x == 'burst0', int(re.findall(r'\d+', x)[0].lstrip('0') or '0'), x))
    # create a list of image objects from the PNG files
    images = [Image.open(path + f) for f in files]
    # create an animated GIF from the list of images
    images[0].save(f'{path}/animation.gif', save_all=True, append_images=images[1:], duration=300, loop=0)






def nearest(items, pivot):
    """
    This function returns the object in 'items' that is the closest to the object 'pivot'.
    """
    found = min(items, key=lambda x: abs(x - pivot))
    return found




def custom_formatter(x, pos):
    """
    Define custom tick formatter for y-axis
    """
    return f'{x:.2f}'



def minmax_normalize(arr=None):
    """
    Min-Max normalization.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    arr_norm = (arr - min_val) / (max_val - min_val)
    return arr_norm




def find_links(url='http://www.python.org'):
    """
    Get all the links in a webpage. 
    Source: https://stackoverflow.com/questions/20150184/make-a-list-of-all-the-files-on-a-website 
    """
    soup = BeautifulSoup(requests.get(url).text)
    hrefs = []
    for a in soup.find_all('a'):
        hrefs.append(a['href'])
    return hrefs




def split_date(dt=None):
    """
    Split a datetime object into year, month, and day, and return them as strings.
        dt: Datetime/Timestamp object.
    """
    YEAR = dt.date().year
    MONTH = dt.date().month
    DAY = dt.date().day
    
    if DAY < 10:
        DAY = f'0{DAY}'
    else:
        DAY = f'{DAY}'
    
    if MONTH < 10:
        MONTH = f'0{MONTH}'
    else:
        MONTH = f'{MONTH}'
    
    return YEAR, MONTH, DAY




# def split_datetime(start=None, end=None):
    
#     START_DATE, START_TIME = start.split('T')
#     END_DATE, END_TIME = end.split('T')

#     START_YEAR, START_MONTH, START_DAY = START_DATE.split('-')
#     END_YEAR, END_MONTH, END_DAY = END_DATE.split('-')

#     START_HOUR, START_MINUTE, START_SECOND = START_TIME.split(':')
#     END_HOUR, END_MINUTE, END_SECOND = END_TIME.split(':')

#     datetime_dict = {
#         'start_year': START_YEAR,
#         'start_month': START_MONTH,
#         'start_day': START_DAY,
#         'start_hour': START_HOUR,
#         'start_minute': START_MINUTE,
#         'start_second': START_SECOND,
        
#         'end_year': END_YEAR,
#         'end_month': END_MONTH,
#         'end_day': END_DAY,
#         'end_hour': END_HOUR,
#         'end_minute': END_MINUTE,
#         'end_second': END_SECOND
#     }
#     return datetime_dict







def fetch_aia(data_dir=None, start=None, end=None, channel=193):
    aia_result = Fido.search(a.Time(start, end),
                             a.Instrument('AIA'),
                             a.Wavelength(channel*u.angstrom),
                             a.Sample(1*u.min))
    aia_files = Fido.fetch(aia_result, path=data_dir)
    print('AIA data is fetched sccessfully')
    return aia_files








def load_aia(data_dir=None, start=None, end=None, level=1.5, data_type='highres', promote=False, channel=193):
    
    # Check if the datetime is a string
    if isinstance(start, str) and isinstance(end, str):
        dt_dict = split_datetime(start=start, end=end)
    
    # Check if the datetime is a pandas.Timestamp
    elif isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
        dt_dict = split_datetime(start=str(start).replace(' ','T'), end=str(end).replace(' ','T'))
    
    if level == 1.5:
        data_path = f'{data_dir}/AIA/{channel}A/{data_type}/lv15'
    else:
        data_path = f'{data_dir}/AIA/{channel}A/{data_type}/lv1'
    
    data = sorted(glob.glob(f'{data_path}/aia*{channel}A_*.fits'))
    
    start_pattern = f"aia*{channel}A_{dt_dict['start_year']}_{dt_dict['start_month']}_{dt_dict['start_day']}T{dt_dict['start_hour']}_{dt_dict['start_minute']}*"
    end_pattern   = f"aia*{channel}A_{dt_dict['end_year']}_{dt_dict['end_month']}_{dt_dict['end_day']}T{dt_dict['end_hour']}_{dt_dict['end_minute']}*"
    
    first_file_to_find = sorted(glob.glob(f'{data_path}/{start_pattern}*.fits'))
    last_file_to_find  = sorted(glob.glob(f'{data_path}/{end_pattern}*.fits'))
    
    idx1 = data.index(first_file_to_find[0])
    idx2 = data.index(last_file_to_find[0])
    
    chosen_files = data[idx1:idx2]
    
    map_objects = []
    for i, file in enumerate(chosen_files):
        # load the file as a sunpy map
        m = sunpy.map.Map(file)
        print(f'AIA {channel}A image {i} is loaded')

        if level == 1:
            if promote:
                print(f'{i} Upgrade AIA {channel} map to lv1.5')
                # update the metadata of the map to the most recent pointing
                m_updated = update_pointing(m)
                # scale the image to the 0.6"/pix
                # and derotate the image such that the y-axis is aligned with solar North
                m_registered = register(m_updated)
                # exposure time normalization
                m_normalized = m_registered / m_registered.exposure_time
                map_objects.append(m_normalized)
            else:
                print(f'Append lv1 AIA {channel} map {i}')
                map_objects.append(m)
        else:
            print(f'Append lv1.5 AIA {channel} map {i}')
            map_objects.append(m)
    return map_objects




def save_processed_aia(data_dir=None, data=None, channel=193):
    for i, processed_map in enumerate(data):
        text_string = processed_map.meta['date-obs']
        # make translation table and use it
        translation_table = str.maketrans('-:.', '___')
        result = text_string.translate(translation_table)
        output_filename = f'aia_{channel}a_{result}_lev15'
        file_path = f'{data_dir}/AIA/{channel}A/{data_type}/lv15/{output_filename}.fits'
        if not os.path.exists(file_path):
            processed_map.save(file_path, filetype='auto')
            print(f'Image {i} is exported')
        else:
            print(f'Image {i} exists already')




def load_aia_single(data_dir=None, start=None, end=None, level=1, channel=193):
    # Check if the datetime is a string
    if isinstance(start, str) and isinstance(end, str):
        dt_dict = split_datetime(start=start, end=end)
    
    # Check if the datetime is a pandas.Timestamp
    elif isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
        dt_dict = split_datetime(start=str(start).replace(' ','T'), end=str(end).replace(' ','T'))

    if level == 1.5:
        data_path = f'{data_dir}/AIA/{channel}A/lv15'
    else:
        data_path = f'{data_dir}/AIA/{channel}A'
    
    # data = sorted(glob.glob(f'{data_path}/aia_{channel}a_*.fits'))
    data = sorted(glob.glob(f'{data_path}/aia_lev1_{channel}a_*.fits'))
    
    start_filename = f"aia_lev1_{channel}a_{dt_dict['start_year']}_{dt_dict['start_month']}_{dt_dict['start_day']}t{dt_dict['start_hour']}_{dt_dict['start_minute']}"
    end_filename   = f"aia_lev1_{channel}a_{dt_dict['end_year']}_{dt_dict['end_month']}_{dt_dict['end_day']}t{dt_dict['end_hour']}_{dt_dict['end_minute']}"
    
    first_file_to_find = sorted(glob.glob(f'{data_path}/{start_filename}*.fits'))
    last_file_to_find  = sorted(glob.glob(f'{data_path}/{end_filename}*.fits'))
    
    # Check if the lists are non-empty
    if not first_file_to_find:
        print(f"No files found for the start time pattern: {data_path}/{start_filename}")
        return []
    
    if not last_file_to_find:
        print(f"No files found for the end time pattern: {data_path}/{end_filename}")
        return []
    
    try:
        idx1 = data.index(first_file_to_find[0])
        idx2 = data.index(last_file_to_find[0])
    except ValueError as e:
        print(f"File not found in data list: {e}")
        return []  # Handle the error appropriately, perhaps by logging or returning an empty list
    
    chosen_files = data[idx1:idx2]
    
    map_objects = []
    for i, file in enumerate(chosen_files):
        # load the file as a sunpy map
        m = sunpy.map.Map(file)
        map_objects.append(m)
        print(f'AIA {channel}A image {i} is loaded')
    return map_objects




def load_suvi(data_dir=None, start=None, end=None, channel=195):
    """
    * 9.4 nm (FeXVIII)
    * 13.1 nm (FeXXI)
    * 17.1 nm (FeIX/X)
    * 19.5 nm (FeXII)
    * 28.4 nm (FeXV)
    * 30.4 nm (HeII)
    """
    dt_dict = split_datetime(start=start, end=end)
    data_path = f"{data_dir}/SUVI/{dt_dict['start_year']}{dt_dict['start_month']}{dt_dict['start_day']}/{channel}A"
    root_filename = f"dr_suvi-l2-ci{channel}_g18_s"
    data = sorted(glob.glob(f"{data_path}/{root_filename}{dt_dict['start_year']}{dt_dict['start_month']}{dt_dict['start_day']}*.fits"))
    
    start_file_to_find = f"{data_path}/{root_filename}{dt_dict['start_year']}{dt_dict['start_month']}{dt_dict['start_day']}T{dt_dict['start_hour']}0000Z_e{dt_dict['start_year']}{dt_dict['start_month']}{dt_dict['start_day']}T{dt_dict['start_hour']}0400Z_v1-0-2.fits"
    end_file_to_find   = f"{data_path}/{root_filename}{dt_dict['end_year']}{dt_dict['end_month']}{dt_dict['end_day']}T{dt_dict['end_hour']}2800Z_e{dt_dict['end_year']}{dt_dict['end_month']}{dt_dict['end_day']}T{dt_dict['end_hour']}3200Z_v1-0-2.fits"
    
    idx1 = data.index(start_file_to_find)
    idx2 = data.index(end_file_to_find)
    
    chosen_files = data[idx1:idx2]
    
    map_objects = []
    for i, file in enumerate(chosen_files):
        m = suvi.files_to_map(file, despike_l1b=True)
        min_range = 0
        if channel == 94:
            max_range = 20
        elif channel == 171:
            max_range = 20
        elif channel == 131:
            max_range = 20
        elif channel == 195:
            max_range = 50
        elif channel == 284:
            max_range = 50
        elif channel == 304:
            max_range = 100
        
        m.plot_settings['norm'] = ImageNormalize(vmin=min_range, vmax=max_range, stretch=LogStretch())
        map_objects.append(m)
        print(f'SUVI image {i} is done')
    return map_objects



# def load_lasco(data_dir=None, start=None, end=None, detector='C2'):
#     """
#     Load SOHO/LASCO C2 or C3 images as sunpy maps.
#     """
#     dt_dict = split_datetime(start=start, end=end)
#     data = sorted(glob.glob(f"{data_dir}/LASCO_{detector}/LASCO_{detector}_{dt_dict['start_year']}{dt_dict['start_month']}{dt_dict['start_day']}*.jp2"))
    
#     start_file_to_find = f"{data_dir}/LASCO_{detector}/LASCO_{detector}_{dt_dict['end_year']}{dt_dict['start_month']}{dt_dict['start_day']}T{dt_dict['start_hour']}{dt_dict['start_minute']}.jp2"
#     end_file_to_find = f"{data_dir}/LASCO_{detector}/LASCO_{detector}_{dt_dict['end_year']}{dt_dict['end_month']}{dt_dict['end_day']}T{dt_dict['end_hour']}{dt_dict['end_minute']}.jp2"
    
#     idx1 = data.index(start_file_to_find)
#     idx2 = data.index(end_file_to_find)
#     chosen_files = data[idx1:idx2]
    
#     map_objects = []
#     for i, file in enumerate(chosen_files):
#         m = sunpy.map.Map(file)
#         m.meta['bunit'] = 'ct' # a workaround for C2 and C3 jp2 images
#         m.plot_settings['norm'] = ImageNormalize(vmin=0, vmax=250)
#         map_objects.append(m)
#         print(f'LASCO {detector} image {i} is done')
#     return map_objects



def remove_redundant_maps(maps):
    """
    Remove redundant SunPy maps, keeping only one map per unique timestamp.

    Parameters:
    maps (list): List of SunPy Map objects. Each map is expected to have a 'date-obs' 
                 key in its metadata that provides the observation timestamp.

    Returns:
    list: A list of unique SunPy Map objects, one per unique timestamp.
    
    Example:
    >>> unique_maps = remove_redundant_maps(list_of_sunpy_maps)
    """
    unique_maps = {}
    for m in maps:
        timestamp = m.latex_name
        if timestamp not in unique_maps:
            unique_maps[timestamp] = m
    return list(unique_maps.values())



# def apply_runratio(maps):
#     """
#     Apply running-ratio image technique on EUV images.
#     See: https://iopscience.iop.org/article/10.1088/0004-637X/750/2/134/pdf
#         Inputs: list of EUV sunpy maps.
#         Output: sequence of run-ratio sunpy maps.
#     """
#     runratio = [m / prev_m.quantity for m, prev_m in zip(maps[1:], maps[:-1])]
#     m_seq_runratio = sunpy.map.Map(runratio, sequence=True)
    
#     # for m in m_seq_runratio:
#     #     m.data[np.isnan(m.data)] = 1
#     #     m.plot_settings['norm'] = colors.Normalize(vmin=0, vmax=2)
#     #     m.plot_settings['cmap'] = 'Greys_r'
    
#     return m_seq_runratio



def enhance_contrast(image, vmin, vmax):
    """
    Enhance contrast by clipping and normalization.
    """
    image_clipped = np.clip(image, vmin, vmax)
    image_normalized = (image_clipped - vmin) / (vmax - vmin)
    return image_normalized



def calculate_percentiles(image, lower_percentile=3, upper_percentile=97):
    """
    Calculate vmin and vmax based on the 1st and 99th percentiles.
    """
    vmin = np.percentile(image, lower_percentile)
    vmax = np.percentile(image, upper_percentile)
    return vmin, vmax





def geo_to_cartesian(latitude, longitude, radius=1):
    """
    Convert geographic coordinates to Cartesian coordinates.
    """
    # Convert degrees to radians
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)
    
    # Cartesian coordinates
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return x, y, z





def fetch_xrs(data_dir=None, dt=None, year=None, month=None, day=None):
    """
    Download GOES/XRS data.
        dt: Datetime/Timestamp object.
        Inputs: year, month, day.
    """
    try:
        os.makedirs(f'{data_dir}/XRS/', exist_ok=True)
    except:
        pass
    
    if year == 2019:
        # Check if the file doesn't exist
        file_path = f'{data_dir}/XRS/sci_xrsf-l2-flx1s_g17_d{year}{month}{day}_v2-2-0.nc'

        if not os.path.exists(file_path):
            # get the data file
            URL = f'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes17/l2/data/xrsf-l2-flx1s_science/{year}/{month}/sci_xrsf-l2-flx1s_g17_d{year}{month}{day}_v2-2-0.nc'
            response = requests.get(URL)
            open(os.path.join(f'{data_dir}/XRS', f'sci_xrsf-l2-flx1s_g17_d{year}{month}{day}_v2-2-0.nc'), 'wb').write(response.content)

        goes = ts.TimeSeries(file_path)
    else:
        file_path = f'{data_dir}/XRS/sci_xrsf-l2-avg1m_g16_d{year}{month}{day}_v2-2-0.nc'
        
        if not os.path.exists(file_path):
            goes_result = Fido.search(a.Time(f'{dt.date()}', f'{dt.date()}'), a.Instrument('XRS'), a.goes.SatelliteNumber(16))
            goes_file = Fido.fetch(goes_result, path=f'{data_dir}/XRS/')
            goes = ts.TimeSeries(goes_file)
        else:
            goes = ts.TimeSeries(file_path)
    return goes






def fetch_waves(data_dir=None, dt=None, year=None, month=None, day=None):
    """
    Download Wind/WAVES data.
        dt: Datetime/Timestamp object.
        Inputs: year, month, day.
    """
    try:
        os.makedirs(f'{data_dir}/wind_data/', exist_ok=True)
    except:
        pass
    
    file_path = f'{data_dir}/wind_data/waves/wav_h1/{year}/wi_h1_wav_{year}{month}{day}_v01.cdf'
    
    if not os.path.exists(file_path):
        print('Wind/WAVES data file not exist!')
        print('Proceed to fetch data from the server ..')
        
        # get the data file
        URL = f'https://spdf.gsfc.nasa.gov/pub/data/wind/waves/wav_h1/{year}/wi_h1_wav_{year}{month}{day}_v01.cdf'
        response = requests.get(URL)
        open(os.path.join(f'{data_dir}/wind_data/waves/wav_h1/{year}', f'wi_h1_wav_{year}{month}{day}_v01.cdf'), 'wb').write(response.content)
        
        print('Wind/WAVES data is downloaded locally.')
        wind = pycdf.CDF(f'{data_dir}/wind_data/waves/wav_h1/{year}/wi_h1_wav_{year}{month}{day}_v01.cdf')
    else:
        print('Wind/WAVES data exists locally.')
        wind = pycdf.CDF(f'{data_dir}/wind_data/waves/wav_h1/{year}/wi_h1_wav_{year}{month}{day}_v01.cdf')
        
    wind_time = [mdates.date2num(tm) for tm in pd.to_datetime(wind.get('Epoch'))]
    RAD1_freq = np.array(wind.get('Frequency_RAD1'))
    RAD2_freq = np.array(wind.get('Frequency_RAD2'))
    TNR_freq = np.array(wind.get('Frequency_TNR'))
    RAD1_int = np.array(wind.get('E_VOLTAGE_RAD1'))
    RAD2_int = np.array(wind.get('E_VOLTAGE_RAD2'))
    TNR_int = np.array(wind.get('E_VOLTAGE_TNR'))

    wind_freq = np.concatenate((TNR_freq, RAD1_freq, RAD2_freq))
    wind_data = np.concatenate((TNR_int, RAD1_int, RAD2_int), axis=1)

    # Apply Gaussian smoothing
    smoothed_wind = gaussian_filter(wind_data, sigma=1)
    wind_norm = ImageNormalize(smoothed_wind, interval=PercentileInterval(97), clip=True)
    return wind_time, wind_freq, smoothed_wind, wind_norm




def fetch_STAswaves(data_dir=None, year=None, month=None, day=None):
    """
    Download STEREO/SWAVES data.
        Inputs: year, month, day.
    """
    try:
        os.makedirs(f'{data_dir}/stereo_data/', exist_ok=True)
    except:
        pass
    
    file_path = f'{data_dir}/stereo_data/stereo_level2_swaves_{year}{month}{day}_v02.cdf'

    if not os.path.exists(file_path):
        # get the data file
        URL = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/combined/swaves/level2_cdf/{year}/stereo_level2_swaves_{year}{month}{day}_v02.cdf'
        response = requests.get(URL)
        open(os.path.join(f'{data_dir}/stereo_data', f'stereo_level2_swaves_{year}{month}{day}_v02.cdf'), 'wb').write(response.content)
    
    # read the data values
    cdf_stereo = pycdf.CDF(f'{data_dir}/stereo_data/stereo_level2_swaves_{year}{month}{day}_v02.cdf')
    time_ste = np.array(cdf_stereo.get('Epoch'))
    freq_ste = np.array(cdf_stereo.get('frequency'))
    data_ste_A = np.array(cdf_stereo.get('avg_intens_ahead'))

    # calculate the mean value in each row (freq channel)
    # and subtract it from each corresponding row
    df_ste_A = pd.DataFrame(data_ste_A.T)
    df_ste_mean = df_ste_A.mean(axis=1)
    data_ste_A = df_ste_A.sub(df_ste_mean, axis=0)

    # Apply Gaussian smoothing
    smoothed_ste_A = gaussian_filter(data_ste_A, sigma=1)
    ste_norm = ImageNormalize(smoothed_ste_A, interval=PercentileInterval(97), clip=True)
    return time_ste, freq_ste, smoothed_ste_A, ste_norm





def fetch_PSPfields(data_dir=None, year=None, month=None, day=None, data_version=2):
    """
    Download PSP/FIELDS data.
        Inputs: year, month, day.
    """
    try:
        os.makedirs(f'{data_dir}/psp_data/', exist_ok=True)
    except:
        pass
    
    hfr_file_path = f'psp_fld_l2_rfs_hfr_{year}{month}{day}_v0{data_version}.cdf'
    lfr_file_path = f'psp_fld_l2_rfs_lfr_{year}{month}{day}_v0{data_version}.cdf'
    
    HFR_URL = f'http://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/rfs_hfr/{year}/{month}/{hfr_file_path}'
    LFR_URL = f'http://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/rfs_lfr/{year}/{month}/{lfr_file_path}'

    hfr_response = requests.head(HFR_URL)
    lfr_response = requests.head(LFR_URL)

    if hfr_response.status_code == 200 and lfr_response.status_code == 200:
        # Files are available, proceed with download
        if not os.path.exists(hfr_file_path):
            response = requests.get(HFR_URL)
            open(os.path.join(f'{data_dir}/psp_data', hfr_file_path), 'wb').write(response.content)
        
        if not os.path.exists(lfr_file_path):
            response = requests.get(LFR_URL)
            open(os.path.join(f'{data_dir}/psp_data', lfr_file_path), 'wb').write(response.content)
        
        # load the PSP data
        cdf_psp_hfr = pycdf.CDF(os.path.join(f'{data_dir}/psp_data', hfr_file_path))
        cdf_psp_lfr = pycdf.CDF(os.path.join(f'{data_dir}/psp_data', lfr_file_path))

        # the min power scaled power spectral density (PSD) of 1e-16 is used as a threshold
        # convert pixels values to dB
        arr_lfr = np.array(cdf_psp_lfr.get('psp_fld_l2_rfs_lfr_auto_averages_ch0_V1V2'))
        Lp_lfr = 10*np.log10(arr_lfr/10**-16) # z-axis
        tm_lfr = np.array(cdf_psp_lfr.get('epoch_lfr')) # x-axis
        freq_lfr = np.array(cdf_psp_lfr.get('frequency_lfr_auto_averages_ch0_V1V2'))/10**6 # y-axis

        # convert pixels values to dB
        arr_hfr = np.array(cdf_psp_hfr.get('psp_fld_l2_rfs_hfr_auto_averages_ch0_V1V2'))
        Lp_hfr = 10*np.log10(arr_hfr/10**-16) # z-axis
        tm_hfr = np.array(cdf_psp_hfr.get('epoch_hfr')) # x-axis
        freq_hfr = np.array(cdf_psp_hfr.get('frequency_hfr_auto_averages_ch0_V1V2'))/10**6 # y-axis

        ### clean the dyspec by subtracting the Mean intensity from each freq channel
        df_psp_hfr = pd.DataFrame(Lp_hfr.T)
        df_psp_lfr = pd.DataFrame(Lp_lfr.T)

        # concat the two arrays of both bands
        df_lfr = pd.DataFrame(df_psp_lfr)
        df_lfr.insert(loc=0, column='frequency', value=freq_lfr[0])
        df_lfr.set_index(['frequency'], inplace=True)

        df_hfr = pd.DataFrame(df_psp_hfr)
        df_hfr.insert(loc=0, column='frequency', value=freq_hfr[0])
        df_hfr.set_index(['frequency'], inplace=True)

        # drop the overlapped rows, take only the first row of the duplicated group
        df_psp = pd.concat([df_lfr, df_hfr])
        df_psp = df_psp.sort_index(axis=0)
        df_psp = df_psp[~df_psp.index.duplicated(keep='first')]
        df_psp.columns = max([tm_lfr, tm_hfr], key=len)
        
        # calculate the mean value in each row (freq channel)
        # and subtract it from each corresponding row
        df_psp_mean = df_psp.mean(axis=1)
        df_psp_submean = df_psp.sub(df_psp_mean, axis=0)
        
        # Check for NaN and -inf values and replace them with zero
        df_psp_submean.replace([np.nan, -np.inf], 0, inplace=True)

        # Apply Gaussian smoothing
        psp_submean_smooth = gaussian_filter(df_psp_submean, sigma=1)
        df_smoothed_psp = pd.DataFrame(psp_submean_smooth, index=df_psp.index, columns=df_psp.columns)
        psp_norm = ImageNormalize(psp_submean_smooth, interval=PercentileInterval(97), clip=True)
        return df_smoothed_psp, psp_norm
    else:
        # Files are not available
        print('PSP/FIELD data is not available on the website!\n')
        return None




# def draw_bezier(x1=0, y1=0, x2=0, y2=0, control=[0,0]):
#     """
#     Draw a Bezier curve using the given control points.
#     The curve will be drawn from the point (x1, y1) to the point
#     (x2, y2) using the control points (control[0], control[1]).
#     """
#     A = np.array([x2, y2])
#     B = np.array(control)
#     C = np.array([x1, y1])

#     A = A.reshape(2,1)
#     B = B.reshape(2,1)
#     C = C.reshape(2,1)
    
#     t = np.arange(0, 1, 0.2).reshape(1,-1)
    
#     # length = len(df.index)
#     # t = np.linspace(0, 1, length).reshape(1,-1)
#     # t = np.arange(0, length, 1).reshape(1,-1)
    
#     P0 = A * t + (1 - t) * B
#     P1 = B * t + (1 - t) * C
#     Pfinal = P0 * t + (1 - t) * P1

#     return Pfinal


# def extract_bezier_values(array, x1, y1, x2, y2, control):
#     """
#     Extract the values of a Bezier curve at the given control points.
#     The curve will be drawn from the point (x1, y1) to the point
#     (x2, y2) using the control points (control[0], control[1])
#     """
#     Pfinal = draw_bezier(x1, y1, x2, y2, control)
#     x_coords = np.round(Pfinal[0, :]).astype(int)
#     y_coords = np.round(Pfinal[1, :]).astype(int)

#     # Clip the coordinates to stay within array bounds
#     x_coords = np.clip(x_coords, 0, array.shape[1] - 1)
#     y_coords = np.clip(y_coords, 0, array.shape[0] - 1)

#     # Extract values along the Bézier curve
#     bezier_values = array[y_coords, x_coords]
    
#     return bezier_values, x_coords, y_coords



# def extract_bezier_values(array, x1, y1, x2, y2, control):
#     """
#     Extract the values of a Bezier curve at the given control points.
#     The curve will be drawn from the point (x1, y1) to the point
#     (x2, y2) using the control points (control[0], control[1])
#     """
#     Pfinal = draw_bezier(x1, y1, x2, y2, control)
#     x_coords = Pfinal[0, :]  # Keep as floating-point
#     y_coords = Pfinal[1, :]  # Keep as floating-point

#     # Clip the coordinates to stay within array bounds
#     x_coords_int = np.clip(np.round(x_coords).astype(int), 0, array.shape[1] - 1)
#     y_coords_int = np.clip(np.round(y_coords).astype(int), 0, array.shape[0] - 1)

#     # Extract values along the Bézier curve
#     bezier_values = array[y_coords_int, x_coords_int]
    
#     return bezier_values, x_coords, y_coords



def generate_number_list(center, offset, count):
    """
    Generate a list of numbers around a given center number.

    Parameters:
    center (int or float): The central number of the list.
    offset (int or float): The increment by which numbers are spaced from the center.
    count (int): The number of numbers to include on each side of the center number.

    Returns:
    list: A list of numbers around the given center.

    Example usage:
        center_number = 0
        offset_value = 2
        count_around = 3
        number_list = generate_number_list(center_number, offset_value, count_around)
        print(number_list)
        [-6, -4, -2, 0, 2, 4, 6]
    """
    return [center + offset * i for i in range(-count, count + 1)]



def compute_standard_error(values_list):
    values_array = np.array(values_list)
    mean_values = np.mean(values_array, axis=0)
    standard_error = np.std(values_array, axis=0, ddof=1) / np.sqrt(values_array.shape[0])
    return mean_values, standard_error





def draw_bezier(x1=0, y1=0, x2=0, y2=0, controls=[[0,0]], n=2):
    """
    Draw a Bézier curve of degree n using control points.
    
    Parameters:
    - x1, y1: Start point coordinates.
    - x2, y2: End point coordinates.
    - controls: A list of control points, where:
        - 1 control point for n=2 (quadratic).
        - 2 control points for n=3 (cubic).
    - n: Degree of the Bézier curve (2 for quadratic, 3 for cubic).
    
    Returns:
    - bezier_curve: An array of points [x, y] that form the Bézier curve.
    """
    if n == 2 and len(controls) != 1:
        raise ValueError("Quadratic Bézier requires exactly 1 control point.")
    elif n == 3 and len(controls) != 2:
        raise ValueError("Cubic Bézier requires exactly 2 control points.")
    
    # Convert points to numpy arrays
    P0 = np.array([x1, y1])  # Start point
    P3 = np.array([x2, y2])  # End point
    
    # Create time steps t from 0 to 1
    t = np.linspace(0, 1, 50)  # 100 points for smoothness

    if n == 2:  # Quadratic Bézier curve
        P1 = np.array(controls[0])  # Only 1 control point
        
        # Quadratic Bézier formula
        bezier_curve = (1 - t)[:, None] ** 2 * P0 + \
                       2 * (1 - t)[:, None] * t[:, None] * P1 + \
                       t[:, None] ** 2 * P3
    
    elif n == 3:  # Cubic Bézier curve
        if not any(np.isnan(controls[1])):
            P1 = np.array(controls[0])  # First control point
            P2 = np.array(controls[1])  # Second control point
            
            # Cubic Bézier formula
            bezier_curve = (1 - t)[:, None] ** 3 * P0 + \
                           3 * (1 - t)[:, None] ** 2 * t[:, None] * P1 + \
                           3 * (1 - t)[:, None] * t[:, None] ** 2 * P2 + \
                           t[:, None] ** 3 * P3
        else:
            P1 = np.array(controls[0])  # Only 1 control point
            
            # Quadratic Bézier formula
            bezier_curve = (1 - t)[:, None] ** 2 * P0 + \
                           2 * (1 - t)[:, None] * t[:, None] * P1 + \
                           t[:, None] ** 2 * P3
    
    return bezier_curve



def extract_bezier_values(array, x1, y1, x2, y2, controls, n):
    """
    Extract the values of a Bézier curve of degree n using control points.
    The curve will be drawn from the point (x1, y1) to the point (x2, y2)
    using the control points provided in the controls list.
    """
    bezier_curve = draw_bezier(x1, y1, x2, y2, controls, n)
    
    # Get the x and y coordinates
    x_coords = np.round(bezier_curve[:, 0]).astype(int)
    y_coords = np.round(bezier_curve[:, 1]).astype(int)

    # Clip the coordinates to stay within array bounds
    x_coords = np.clip(x_coords, 0, array.shape[1] - 1)
    y_coords = np.clip(y_coords, 0, array.shape[0] - 1)

    # Extract values along the Bézier curve
    bezier_values = array[y_coords, x_coords]
    
    return bezier_values, x_coords, y_coords

















