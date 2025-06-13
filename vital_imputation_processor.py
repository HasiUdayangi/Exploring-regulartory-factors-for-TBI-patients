
"""
vital sign imputation
Hasitha
"""


import pandas as pd
import boto3
import sys
import numpy as np

from utils.waveform_viewer2 import Waveform_Helper
from utils import Athena_Query
from utils import S3, LabelStore
from utils.waveform_viewer2 import Waveform_Chart
from utils.waveform_viewer2 import Waveform_Extract, Waveform_Helper
import datetime
from datetime import datetime, timedelta, date
from vital_sign_extraction import VitalSignExtractor
vital = VitalSignExtractor()

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer


class HealthDataProcessor:
    def __init__(self):
        self.vital = VitalSignExtractor()
        self.athena = Athena_Query()
    
    def every_minutes_dataframe1(self, patient_ids, parameter_ids, admission_date, whole_stay=False):
        if whole_stay:
            df2 = self.vital.all_vital_extract(patient_ids, parameter_ids, admission_date)
        else:
            df2 = self.vital.extract_12hr_vital(patient_ids, parameter_ids, admission_date)
        
        df2['time'] = pd.to_datetime(df2['time'])
        time_range = pd.date_range(start=df2['time'].min(), end=df2['time'].max(), freq='1T')
        df_minutes = pd.DataFrame({'time': time_range})
    
        df_minutes = df_minutes.merge(df2, on='time', how='left')
        df_minutes['time_since_admission_minutes'] = df_minutes['time_since_admission'] * 60
        df_minutes['time_since_admission_minutes'] = pd.to_numeric(df_minutes['time_since_admission_minutes'])
    
        last_value = df_minutes['time_since_admission_minutes'].last_valid_index()
        for index, row in df_minutes.iterrows():
            if pd.notna(row['time_since_admission_minutes']):
                last_value = row['time_since_admission_minutes']
            else:
                last_value += 1
            df_minutes.at[index, 'time_since_admission_minutes'] = last_value
    
        df_minutes['time_since_admission_hours'] = df_minutes['time_since_admission_minutes'] / 60.0
        df_minutes.drop(columns=['time_since_admission', 'time_since_admission_minutes'], inplace=True)
    
        return df_minutes
    
    
    def process_health_data1(self, df_min, imputer_type='knn'):
        if imputer_type == 'iterative':
            imputer = IterativeImputer()
        else:
            imputer = KNNImputer()

        null_percentages = df_min.isnull().sum() / len(df_min)
        # Drop columns with null value percentages greater than 0.5
        columns_to_drop = null_percentages[null_percentages > 0.5].index
        df_min_filtered = df_min.drop(columns=columns_to_drop)

        x_ticks = range(1, len(df_min_filtered) + 1)
        x_labels = [f'{minute}' for minute in x_ticks]

        # Reset the index of df_min and drop unnecessary column
        df_allbefore = df_min_filtered.drop(['time_since_admission_hours'], axis=1)

        # Separate the 'time' column from numerical features
        datetime_column = df_allbefore['time']
        numerical_columns = df_allbefore.drop(columns=['time'])

        # Fit the imputer (exclude the 'time_since_admission_hours' column from imputation)
        imputed_data = imputer.fit_transform(numerical_columns)
        imputed_df = pd.DataFrame(imputed_data, columns=numerical_columns.columns)

        df_after = pd.concat([datetime_column, imputed_df], axis=1)
    
        return df_allbefore, df_after, x_ticks, x_labels
    
    def plot_health_parameters_subplots(self, x_ticks, df_allbefore, df_allafter, columns_to_plot, colors, patient_id):
        fig = make_subplots(rows=len(columns_to_plot), cols=1, subplot_titles=columns_to_plot)

        for i, column in enumerate(columns_to_plot):
            try:
                imputed_all_hr = df_allafter[column]
                nan_hr_mask = df_allbefore[column].isnull()
                imputed_hr = imputed_all_hr[nan_hr_mask]
                imputed_x_ticks = list(x_ticks[i] for i in range(len(x_ticks)) if nan_hr_mask[i])

                fig.add_trace(go.Scatter(x=list(x_ticks), y=df_allbefore[column], mode='markers+lines', name=f'Original {column}', line=dict(color=colors[i])),
                              row=i+1, col=1)

                fig.add_trace(go.Scatter(x=imputed_x_ticks, y=imputed_hr, mode='markers', name=f'Imputed {column}', marker=dict(color='red', size=10)),
                              row=i+1, col=1)

                fig.update_xaxes(title_text='Minutes', row=i+1, col=1)
                fig.update_yaxes(title_text='Values', row=i+1, col=1)

            except KeyError:
                pass  # Skip this column if it's not present

        fig.update_layout(showlegend=False, height=2000, title_text=f'Health Parameters for {patient_id} from {df_allbefore["time"].min()} to {df_allbefore["time"].max()}')

        return fig
    
    #whole data
    def every_minutes_dataframe(self, patient_ids, parameter_ids, admission_date):
        
        df2 = self.vital.all_vital_extract(patient_ids, parameter_ids, admission_date)
        df2['time'] = pd.to_datetime(df2['time'])
        time_range = pd.date_range(start=df2['time'].min(), end=df2['time'].max(), freq='1T')
        df_minutes = pd.DataFrame({'time': time_range})

        df_minutes = df_minutes.merge(df2, on='time', how='left')
        df_minutes['time_since_admission_minutes'] = df_minutes['time_since_admission'] * 60
        df_minutes['time_since_admission_minutes'] = pd.to_numeric(df_minutes['time_since_admission_minutes'])

        last_value = df_minutes['time_since_admission_minutes'].last_valid_index()
        for index, row in df_minutes.iterrows():
            if pd.notna(row['time_since_admission_minutes']):
                last_value = row['time_since_admission_minutes']
            else:
                last_value += 1
            df_minutes.at[index, 'time_since_admission_minutes'] = last_value

        df_minutes['time_since_admission_hours'] = df_minutes['time_since_admission_minutes'] / 60.0
        df_minutes.drop(columns=['time_since_admission', 'time_since_admission_minutes'], inplace=True)

        return df_minutes
    
    
    #12 hrs data
    def every_minutes_data(self, patient_ids, parameter_ids, admission_date):
        df2 = self.vital.extract_12hr_vital_new(patient_ids, parameter_ids, admission_date)
        df2['time'] = pd.to_datetime(df2['time'])
        time_range = pd.date_range(start=df2['time'].min(), end=df2['time'].max(), freq='1T')
        df_minutes = pd.DataFrame({'time': time_range})

        df_minutes = df_minutes.merge(df2, on='time', how='left')
        df_minutes['time_since_admission_minutes'] = df_minutes['time_since_admission'] * 60
        df_minutes['time_since_admission_minutes'] = pd.to_numeric(df_minutes['time_since_admission_minutes'])

        last_value = df_minutes['time_since_admission_minutes'].last_valid_index()
        for index, row in df_minutes.iterrows():
            if pd.notna(row['time_since_admission_minutes']):
                last_value = row['time_since_admission_minutes']
            else:
                last_value += 1
            df_minutes.at[index, 'time_since_admission_minutes'] = last_value

        df_minutes['time_since_admission_hours'] = df_minutes['time_since_admission_minutes'] / 60.0
        df_minutes.drop(columns=['time_since_admission', 'time_since_admission_minutes'], inplace=True)

        return df_minutes
    
    
    def process_health_data(self, df_min, imputer):
        null_percentages = df_min.isnull().sum() / len(df_min)
        # Drop columns with null value percentages greater than 0.5
        columns_to_drop = null_percentages[null_percentages > 0.5].index
        df_min_filtered = df_min.drop(columns=columns_to_drop)

        imputer = KNNImputer()
        x_ticks = range(1, len(df_min_filtered) + 1)
        x_labels = [f'{minute}' for minute in x_ticks]

        # Reset the index of df_min and drop unnecessary column
        df_allbefore = df_min_filtered.drop(['time_since_admission_hours'], axis=1)

        # Separate the 'time' column from numerical features
        datetime = df_allbefore['time']
        numerical = df_allbefore.drop(columns=['time'])

        # Fit the KNN imputer (exclude the 'time_since_admission_hours' column from imputation)
        imputed_alldata = imputer.fit_transform(numerical)
        imputed_alldf = pd.DataFrame(imputed_alldata, columns=numerical.columns)
        df_allafter = pd.concat([datetime, imputed_alldf], axis=1)
    
        return df_allbefore, df_allafter, x_ticks, x_labels
    
    
    
    def plot_health_parameters_subplots(self, x_ticks, df_allbefore, df_allafter, columns_to_plot, colors, patient_id):
        fig = make_subplots(rows=len(columns_to_plot), cols=1, subplot_titles=columns_to_plot)

        for i, column in enumerate(columns_to_plot):
            try:
                imputed_all_hr = df_allafter[column]
                nan_hr_mask = df_allbefore[column].isnull()
                imputed_hr = imputed_all_hr[nan_hr_mask]
                imputed_x_ticks = list(x_ticks[i] for i in range(len(x_ticks)) if nan_hr_mask[i])

                fig.add_trace(go.Scatter(x=list(x_ticks), y=df_allbefore[column], mode='markers+lines', name=f'Original {column}', line=dict(color=colors[i])),
                          row=i+1, col=1)

                fig.add_trace(go.Scatter(x=imputed_x_ticks, y=imputed_hr, mode='markers', name=f'Imputed {column}', marker=dict(color='red', size=10)),
                          row=i+1, col=1)

                fig.update_xaxes(title_text='Minutes', row=i+1, col=1)
                fig.update_yaxes(title_text='Values', row=i+1, col=1)

            except KeyError:
                pass  # Skip this column if it's not present

        fig.update_layout(showlegend=False, height=2000, title_text=f'Health Parameters for {patient_id} from {df_allbefore["time"].min()} to {df_allbefore["time"].max()}')

        return fig
    
    
    def every_minutes_df(self, df):
        df['time'] = pd.to_datetime(df['time'])
        time_range = pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='1T')
        df_minutes = pd.DataFrame({'time': time_range})

        df_minutes = df_minutes.merge(df, on='time', how='left')
        df_minutes['time_since_start'] = df_minutes['time_since_start'] * 60
        df_minutes['time_since_start'] = pd.to_numeric(df_minutes['time_since_start'])

        last_value = df_minutes['time_since_start'].last_valid_index()
        for index, row in df_minutes.iterrows():
            if pd.notna(row['time_since_start']):
                last_value = row['time_since_start']
            else:
                last_value += 1
            df_minutes.at[index, 'time_since_start'] = last_value

        df_minutes['time_since_start'] = df_minutes['time_since_start'] / 60.0
        #df_minutes.drop(columns=['time_since_start', 'time_since_start'], inplace=True)

        return df_minutes

    
    
    

    
    
