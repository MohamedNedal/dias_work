#!/usr/bin/env python3
# coding: utf-8

'''
This script calculates the difference in seconds between two datetime strings provided as command-line arguments.
'''

import argparse
from datetime import datetime



def calculate_seconds_between(datetime_str1, datetime_str2):
    
    # Parse the input strings into datetime objects
    datetime1 = datetime.strptime(datetime_str1, '%Y-%m-%d %H:%M:%S')
    datetime2 = datetime.strptime(datetime_str2, '%Y-%m-%d %H:%M:%S')
    
    # Calculate the difference between the two datetime objects
    time_difference = datetime2 - datetime1
    
    # Get the difference in seconds
    seconds_difference = time_difference.total_seconds()
    
    return seconds_difference



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the difference in seconds between two datetime strings.')
    parser.add_argument('--start', required=True, help='The start datetime in the format "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--end', required=True, help='The end datetime in the format "YYYY-MM-DD HH:MM:SS"')
    
    args = parser.parse_args()
    
    start_datetime = args.start
    end_datetime = args.end
    
    seconds_difference = calculate_seconds_between(start_datetime, end_datetime)
    
    print(f'The difference in seconds is: {seconds_difference}')
