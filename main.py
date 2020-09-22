from fastapi import FastAPI
from typing import List

from joblib import load
import numpy as np

from pydantic import BaseModel


class Data(BaseModel):
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    max_glu_serum: int
    A1Cresult: int
    metformin: int
    repaglinide: int
    nateglinide: int
    chlorpropamide: int
    glimepiride: int
    acetohexamide: int
    glipizide: int
    glyburide: int
    tolbutamide: int
    pioglitazone: int
    rosiglitazone: int
    acarbose: int
    miglitol: int
    troglitazone: int
    tolazamide: int
    insulin: int
    glyburide_metformin: int
    glipizide_metformin: int
    metformin_rosiglitazone: int
    metformin_pioglitazone: int
    change: int
    diabetesMed: int
    numchange: int
    race_Asian: int
    race_Caucasian: int
    race_Hispanic: int
    race_Other: int
    age_10_20: int
    age_20_30: int
    age_30_40: int
    age_40_50: int
    age_50_60: int
    age_60_70: int
    age_70_80: int
    age_80_90: int
    age_90_100: int
    medical_specialty_Emergency_Trauma: int
    medical_specialty_Family_GeneralPractice: int
    medical_specialty_Gastroenterology: int
    medical_specialty_InternalMedicine: int
    medical_specialty_Nephrology: int
    medical_specialty_ObstetricsandGynecology: int
    medical_specialty_Orthopedics: int
    medical_specialty_Orthopedics_Reconstructive: int
    medical_specialty_Other: int
    medical_specialty_Psychiatry: int
    medical_specialty_Pulmonology: int
    medical_specialty_Radiologist: int
    medical_specialty_Surgery_Cardiovascular_Thoracic: int
    medical_specialty_Surgery_General: int
    medical_specialty_Surgery_Neuro: int
    medical_specialty_Surgery_Vascular: int
    medical_specialty_Unknow: int
    medical_specialty_Urology: int
    diag_1_Diabetes: int
    diag_1_Digestive: int
    diag_1_Genitourinary: int
    diag_1_Injury: int
    diag_1_Muscoloskeletal: int
    diag_1_Neoplasms: int
    diag_1_Others: int
    diag_1_Respiratory: int
    diag_2_Diabetes: int
    diag_2_Digestive: int
    diag_2_Genitourinary: int
    diag_2_Injury: int
    diag_2_Muscoloskeletal: int
    diag_2_Neoplasms: int
    diag_2_Others: int
    diag_2_Respiratory: int


features = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
       'time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'insulin', 'glyburide_metformin', 'glipizide_metformin',
       'metformin_rosiglitazone', 'metformin_pioglitazone', 'change',
       'diabetesMed', 'numchange', 'race_Asian', 'race_Caucasian',
       'race_Hispanic', 'race_Other', 'age_10_20', 'age_20_30',
       'age_30_40', 'age_40_50', 'age_50_60', 'age_60_70',
       'age_70_80', 'age_80_90', 'age_90_100',
       'medical_specialty_Emergency_Trauma',
       'medical_specialty_Family_GeneralPractice',
       'medical_specialty_Gastroenterology',
       'medical_specialty_InternalMedicine', 'medical_specialty_Nephrology',
       'medical_specialty_ObstetricsandGynecology',
       'medical_specialty_Orthopedics',
       'medical_specialty_Orthopedics_Reconstructive',
       'medical_specialty_Other', 'medical_specialty_Psychiatry',
       'medical_specialty_Pulmonology', 'medical_specialty_Radiologist',
       'medical_specialty_Surgery_Cardiovascular_Thoracic',
       'medical_specialty_Surgery_General', 'medical_specialty_Surgery_Neuro',
       'medical_specialty_Surgery_Vascular', 'medical_specialty_Unknow',
       'medical_specialty_Urology', 'diag_1_Diabetes', 'diag_1_Digestive',
       'diag_1_Genitourinary', 'diag_1_Injury', 'diag_1_Muscoloskeletal',
       'diag_1_Neoplasms', 'diag_1_Others', 'diag_1_Respiratory',
       'diag_2_Diabetes', 'diag_2_Digestive', 'diag_2_Genitourinary',
       'diag_2_Injury', 'diag_2_Muscoloskeletal', 'diag_2_Neoplasms',
       'diag_2_Others', 'diag_2_Respiratory']

# load model
readmission = load('models/NB_pipeline_imbalance_recall.joblib')


# init app
app = FastAPI()


@app.get("/")
def hello():
    return {"message": "https://joserzapata.github.io/"}


@app.post('/predict')
def predict(data: Data):
    # Extract data in correct order
    data_dict = data.dict()
    to_predict = [data_dict[feature] for feature in features]
    dataf = np.array(to_predict)
    prediction = readmission.predict(dataf.reshape(1, -1))
    if prediction[0] == 0:
        result = "No Readmission"
    else:
        result = "Readmission"
    return {"prediction": result}
