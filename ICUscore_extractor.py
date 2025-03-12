"""
lab result extraction
Hasitha
"""

import sys
sys.path.append("/home/ec2-user/SageMaker/")
import pandas as pd
from utils import Athena_Query
from utils import S3, LabelStore
import boto3
import datetime
from datetime import datetime, timedelta, date
from IPython.display import HTML, clear_output
from utils.waveform_viewer2 import Waveform_Chart
from utils.waveform_viewer2 import Waveform_Extract, Waveform_Helper
wh = Waveform_Helper()

class LabDataExtractor:
    def __init__(self):
        self.athena = Athena_Query()
    
    def get_icustext_query(self, patient_ids, parameter_ids):
        patient_ids_string = "(" + ",".join([f"'{id}'" for id in patient_ids]) + ")"
        parameter_ids_string = "(" + ",".join([str(id) for id in parameter_ids]) + ")"

        query = f"""
            SELECT time, textid, parameterid
            FROM metavision_deid_dtm.textsignals
            WHERE parameterid IN {parameter_ids_string} AND patientid IN {patient_ids_string}
            """

        return query
    
    def get_icuscore_value(self, patient_ids, parameter_ids):
        patient_ids_string = "(" + ",".join([f"'{id}'" for id in patient_ids]) + ")"
        parameter_ids_string = "(" + ",".join([str(id) for id in parameter_ids]) + ")"

        query = f"""
            SELECT t.time, pt.value, t.parameterid
            FROM metavision_deid_dtm.textsignals AS t
            JOIN parametertext AS pt ON t.textid = pt.textid
            WHERE t.parameterid IN {parameter_ids_string} AND t.patientid IN {patient_ids_string}
            """

        return query

    def extract_icustext(self, patient_ids, parameter_ids, admission_date):
        query = self.get_icustext_query(patient_ids, parameter_ids)
        df = self.athena.query_as_pandas(query).drop_duplicates().reset_index(drop=True)
        df.sort_values('time', inplace=True)
        df['time_since_admission'] = (pd.to_datetime(df['time']) - datetime.strptime(admission_date, "%Y-%m-%d %H:%M:%S")).dt.total_seconds() / 3600.0
        return df

    def all_icutext_extract(self, patient_ids, parameter_ids, admission_date):
        column_mapping = {
            21654: 'GCS(Eyes)',
            21655: 'GCS(Motor)',
            21656: 'GCS(Verbal)',
        }


        for patient_id in patient_ids:
            query = self.get_icustext_query([patient_id], parameter_ids)
            df_patient = pd.DataFrame(columns=['time'])
            df_query = self.athena.query_as_pandas(query).drop_duplicates().reset_index(drop=True)

            for parameter_id in parameter_ids:
                df_parameter = df_query[df_query['parameterid'] == parameter_id]
                column_name = column_mapping.get(parameter_id, f'textid_{parameter_id}')
                df_patient = pd.merge(df_patient, df_parameter[['time', 'textid']], how='outer', on='time')
                df_patient.rename(columns={'textid': column_name}, inplace=True)

            df_patient.sort_values('time', inplace=True)

            # Calculate 'time since admission' for each 'time' value
            #admission_date = df_trauma.loc[df_trauma['patientid'] == patient_id, 'admissiondate'].iloc[0]
            df_patient['time'] = pd.to_datetime(df_patient['time'])

            # Convert admission_date to datetime format
            admission_date = datetime.strptime(admission_date, "%Y-%m-%d %H:%M:%S")

            # Calculate the time difference in hours
            df_patient['time_since_admission'] = (df_patient['time'] - admission_date).dt.total_seconds() / 3600

            #df_patient['time_since_admission'] = (pd.to_datetime(df_patient['time']) - datetime.strptime(admission_date, "%Y-%m-%d %H:%M:%S")).dt.total_seconds() / 3600.


        return df_patient
    
