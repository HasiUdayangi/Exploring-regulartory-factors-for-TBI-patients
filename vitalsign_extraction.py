import sys
import pandas as pd
from utils import Athena_Query
from utils import S3, LabelStore
import boto3
import datetime
from datetime import datetime, timedelta, date

class VitalSignExtractor:
    def __init__(self):
        self.athena = Athena_Query()
    
    def get_vitalsign_query(self, patient_ids, parameter_ids):
        patient_ids_string = "(" + ",".join([f"'{id}'" for id in patient_ids]) + ")"
        parameter_ids_string = "(" + ",".join([str(id) for id in parameter_ids]) + ")"

        query = f"""
            SELECT time, value, parameterid
            FROM metavision_deid_dtm.signals
            WHERE parameterid IN {parameter_ids_string} AND patientid IN {patient_ids_string}
            """

        return query

    def extract_vital_signs(self, patient_ids, parameter_ids, admission_date):
        query = self.get_vitalsign_query(patient_ids, parameter_ids)
        df = self.athena.query_as_pandas(query).drop_duplicates().reset_index(drop=True)
        df.sort_values('time', inplace=True)
        df['time_since_admission'] = (pd.to_datetime(df['time']) - datetime.strptime(admission_date, "%Y-%m-%d %H:%M:%S.%f")).dt.total_seconds() / 3600.0
        return df

    def all_vital_extract(self, patient_ids, parameter_ids, admission_date):
        column_mapping = {
            3885: 'HR',
            3888: 'Diastolic BP',
            3887: 'Systolic BP',
            5436: 'ABP Mean',
            3951: 'SpO2',
            4083: 'Respiratory rate',
            3976: 'Temperature',
            3910: 'Intra-Cranial Pressure'
        }


        for patient_id in patient_ids:
            query = self.get_vitalsign_query([patient_id], parameter_ids)
            df_query = self.athena.query_as_pandas(query).drop_duplicates().reset_index(drop=True)
        
            df_patient = pd.DataFrame(columns=['time'])

            for parameter_id in parameter_ids:
                df_parameter = df_query[df_query['parameterid'] == parameter_id]
                column_name = column_mapping.get(parameter_id, f'value_{parameter_id}')
                df_patient = pd.merge(df_patient, df_parameter[['time', 'value']], how='outer', on='time')
                df_patient.rename(columns={'value': column_name}, inplace=True)

            df_patient.sort_values('time', inplace=True)
            df_patient['time_since_start'] = df_patient['time'] - df_patient['time'].iloc[0]

            # Convert the time difference to hours
            df_patient['time_since_start'] = df_patient['time_since_start'].dt.total_seconds() / 3600.0


        return df_patient

    
    
    def all_ABG_extract(self, patient_ids, parameter_ids, admission_date):
        column_mapping = {
            8919: 'PaCo2',
            11424: 'Potassium',
            15166: 'Total Hb',
            13053: 'Oxy Haemoglobin',
            13054: 'Oxygen Saturation',
            12347: 'Methaemoglobin',
            8465: 'Carboxyhaemoglobin',
            13670: 'PaO2',
            8523: 'pH',
            15825: 'Chloride',
            12756: 'Sodium',
            15824: 'Glucose',
            7960: 'Calcium (Ionised)',
            8527: 'pO2',
            13188: 'ABG p50',
            27391: 'Specimen Type',
            10654: 'ABG Bicarbonate',
            11763: 'ABG Lactate',
            3997: 'Pulmonary Artery Pressure Mean',
            7676: 'ABG Base Excess',
            6904: 'ABG Anion Gap'
        }


        for patient_id in patient_ids:
            query = self.get_vitalsign_query([patient_id], parameter_ids)
            df_patient = pd.DataFrame(columns=['time'])
            df_query = self.athena.query_as_pandas(query).drop_duplicates().reset_index(drop=True)

            for parameter_id in parameter_ids:
                df_parameter = df_query[df_query['parameterid'] == parameter_id]
                column_name = column_mapping.get(parameter_id, f'value_{parameter_id}')
                df_patient = pd.merge(df_patient, df_parameter[['time', 'value']], how='outer', on='time')
                df_patient.rename(columns={'value': column_name}, inplace=True)

            df_patient.sort_values('time', inplace=True)

            # Calculate 'time since admission' for each 'time' value
            #admission_date = df_trauma.loc[df_trauma['patientid'] == patient_id, 'admissiondate'].iloc[0]
            df_patient['time_since_admission'] = (pd.to_datetime(df_patient['time']) - datetime.strptime(admission_date, "%Y-%m-%d %H:%M:%S.%f")).dt.total_seconds() / 3600.


        return df_patient
    
    
    def extract_12hr_vital(self, patient_ids, parameter_ids, admission_date):
        target_lower=1
        target_upper=13
        df1 = self.all_vital_extract(patient_ids, parameter_ids, admission_date)

        data_within_range = df1.loc[(df1['time_since_admission'] >= target_lower) & (df1['time_since_admission'] <= target_upper)]

        if not data_within_range.empty:
            df_result = data_within_range
        else:
            closest_to_lower = df1.loc[df1['time_since_admission'] >= target_lower, 'time_since_admission'].idxmin()
            closest_to_upper = df1.loc[df1['time_since_admission'] <= target_upper, 'time_since_admission'].idxmax()

            if abs(df1.loc[closest_to_lower, 'time_since_admission'] - target_lower) <= abs(df1.loc[closest_to_upper, 'time_since_admission'] - target_upper):
                start = closest_to_lower
            else:
                start = closest_to_upper

            time_range = pd.Timedelta(hours=12)
            end = start + time_range

            if end >= len(df1):
                df_result = df1.loc[start:]
            else:
                df_result = df1.loc[start:end]

        return df_result
    
    
    def extract_12hr_vital_new(self, patient_ids, parameter_ids, admission_date):
        target_lower = 1
        target_upper = 13
        df1 = self.all_vital_extract(patient_ids, parameter_ids, admission_date)

        if not df1.empty:  # Check if the DataFrame is not empty
            data_within_range = df1.loc[(df1['time_since_admission'] >= target_lower) & (df1['time_since_admission'] <= target_upper)]

            if not data_within_range.empty:
                df_result = data_within_range
            else:
                closest_to_lower = df1.loc[df1['time_since_admission'] >= target_lower, 'time_since_admission'].idxmin()
                closest_to_upper = df1.loc[df1['time_since_admission'] <= target_upper, 'time_since_admission'].idxmax()

                if abs(df1.loc[closest_to_lower, 'time_since_admission'] - target_lower) <= abs(df1.loc[closest_to_upper, 'time_since_admission'] - target_upper):
                    start = closest_to_lower
                else:
                    start = closest_to_upper

                time_range = pd.Timedelta(hours=12)
                end = start + time_range

                if end >= len(df1):
                    df_result = df1.loc[start:]
                else:
                    df_result = df1.loc[start:end]
        else:
            df_result = pd.DataFrame(columns=['time', 'time_since_admission'])  # Create an empty DataFrame

        return df_result
