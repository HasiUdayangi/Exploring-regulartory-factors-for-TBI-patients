
import sys
sys.path.append("/home/ec2-user/SageMaker/")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timedelta, date
from matplotlib.ticker import MultipleLocator
from dateutil.relativedelta import relativedelta, MO

# Importing utilities from the custom package
from utils import Athena_Query, s3, LabelStore
from utils.sql_query import SqlQuery
from IPython.display import clear_output, HTML
import heartpy as hp
from scipy.interpolate import CubicSpline

athena = Athena_Query()


def extract_12hr_ecg(patientid, start, end):
    """
    Extracts ECG data in 4-hour segments between the given start and end times.

    Parameters:
        patientid (str): The patient identifier.
        start (str): Start datetime in the format '%Y-%m-%d %H:%M:%S'.
        end (str): End datetime in the format '%Y-%m-%d %H:%M:%S'.

    Returns:
        pd.DataFrame or None: Combined ECG data if extraction is successful; otherwise, None.
    """
    max_extraction_time = timedelta(hours=4)
    time_increment = timedelta(hours=4)

    current_start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    current_end_time = current_start_time + max_extraction_time
    found_data = False
    ecg_data = []

    # Loop through the extraction period in 4-hour increments
    while current_end_time <= datetime.strptime(end, '%Y-%m-%d %H:%M:%S'):
        try:
            we = Waveform_Extract(patientid)
            start_str = current_start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_str = current_end_time.strftime('%Y-%m-%d %H:%M:%S')
            we.set_extract_time(start_str, end_str)
            ECG = we.get_ecg(cols=['ecg_ii'])
            ecg_data.append(ECG)
            found_data = True
        except Exception as e:
            print(f"No data found between {current_start_time} and {current_end_time}: {e}")

        current_start_time += time_increment
        current_end_time += time_increment

    clear_output(wait=True)
    if found_data:
        print("Data extraction successful.")
    else:
        print("No data found within the specified time period.")
        return None

    combined_ecg_data = pd.concat(ecg_data)
    return combined_ecg_data

# Example: Extract additional bed time data for a set of patients.
# Note: 'trauma' DataFrame should be defined elsewhere with a column 'patientid'.
if __name__ == "__main__":
    # Example snippet for bed time extraction (ensure 'trauma' is defined)
    cols = ['patientid', 'bedname', 'fromtime', 'totime']
    result_df = pd.DataFrame(columns=cols)  # Create an empty DataFrame
    
    # Uncomment and adjust the following block if 'trauma' DataFrame is available
    # for pid in trauma.patientid.to_numpy():
    #     we = Waveform_Extract(patientid=pid)
    #     beds = we.get_all_bed_times()
    #     for i in range(len(beds)):
    #         new_row = pd.DataFrame(data=[[pid, beds['bedname'][i], beds['fromtime'][i], beds['totime'][i]]], columns=cols)
    #         result_df = pd.concat([result_df, new_row], ignore_index=True)
    #     clear_output(wait=True)
    
    # Example usage of the extract_12hr_ecg function
    patientid_example = ''
    start_time = ''
    end_time = ''
    df_ecg = extract_12hr_ecg(patientid_example, start_time, end_time)
    
    if df_ecg is not None:
        print("Extracted ECG Data:")
        print(df_ecg.head())
    else:
        print("ECG extraction returned no data.")
