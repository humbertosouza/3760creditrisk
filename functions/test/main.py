import io
import os
import sys

import pickle
import pandas as pd 
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
# Ignore ConvergenceWarning messages
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
from google.cloud import storage
from google.cloud import datastore
from google.cloud import firestore

def process_request(request):  

  storage_client = storage.Client()
  bucket = storage_client.get_bucket('credit-risk-bucket')
  # Write/overwrite the incoming file
  fileContent = request.files['record']
  blob = bucket.blob('function_code_new_'+fileContent.filename)
  file_content = request.files['record'].read()
  blob.upload_from_string(file_content.decode('utf-8'))

  #b = bytes(mystring, 'utf-8')
  #Cleans and feature engineer the arrived dataset
  clean_input = request_cleanup(file_content.decode('utf-8'))    

  # Load the trained model 
  trained_model = load_model_data(request.files['record'])

  # Runs the model against the requested record to assess
  credit_score = run_model(clean_input, trained_model)

  result = "Approved" if credit_score[0] == 0 else "Declined"

  # Store the request log in Firestore in Datastore mode
  store_log(request, result,str(credit_score[1]))
  
  return '{"score":"'+result+'","default_probability":"'+str(credit_score[1])+'"}'

def store_log(request,result, probability):
 
  record_data = '"cr_request_origin":"' + str(request.remote_addr) + '",' + \
    '"cr_request_headers":"' + str(request.headers) + '",' + \
    '"cr_request_datetime":"' + str(datetime.datetime.now()) + '",' + \
    '"cr_request_environ":"' + str(request.environ)+'"})' 

  client = datastore.Client(namespace='CREDITASSESSMENT')
  key = client.key("Keys")
  logged_user = request.form.get('logged_user')

  # Create an unsaved Entity object, and tell Datastore not to index the
  # `record_data` field
  task = datastore.Entity(key, exclude_from_indexes=["record_data"])

  # Apply new field values and save the task entity to Datastore
  task.update(
    {
      "last_updated": datetime.datetime.now(tz=datetime.timezone.utc),
      "remote_ip": str(request.remote_addr),
      "logged_user": logged_user,
      "record_data": record_data,
      "active": True,
      "result": result,
      "probability": probability,
      "billed": False,
      "id":0
    }
    )
  client.put(task)
     
  return task.key  

def request_cleanup(file_content):
  """ 
    Clean up the input data
  """
   # Read from the file_content string var
  eval_data = io.StringIO(file_content)
  evaluate = pd.read_csv(eval_data, sep=",") 

  # Data Cleaning for a single record
  evaluate['OCCUPATION_TYPE'].value_counts() 
  evaluate['OCCUPATION_TYPE'].unique()
  evaluate['OCCUPATION_TYPE'].replace([ 'Cleaning staff', 'Cooking staff', 'Accountants', 'Medicine staff'
        'Private service staff', 'Security staff','Waiters/barmen staff',
        'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff', 'HR staff'], 'Others', inplace= True)
  evaluate['ORGANIZATION_TYPE'].value_counts()
  evaluate['ORGANIZATION_TYPE'].replace(['School', 'Religion',
       'Other','Electricity', 'Medicine',
       'Transport: type 2',
       'Construction', 'Housing', 'Kindergarten', 'Trade: type 7',
       'Industry: type 11', 'Military', 'Services', 'Security Ministries',
       'Transport: type 4', 'Industry: type 1', 'Emergency', 'Security',
       'Trade: type 2', 'University', 'Transport: type 3', 'Police',
       'Business Entity Type 1', 'Postal', 'Industry: type 4',
       'Agriculture', 'Restaurant', 'Culture', 'Hotel',
       'Industry: type 7', 'Trade: type 3', 'Industry: type 3', 'Bank',
       'Industry: type 9', 'Insurance', 'Trade: type 6',
       'Industry: type 2', 'Transport: type 1', 'Industry: type 12',
       'Mobile', 'Trade: type 1', 'Industry: type 5', 'Industry: type 10',
       'Legal Services', 'Advertising', 'Trade: type 5', 'Cleaning',
       'Industry: type 13', 'Trade: type 4', 'Telecom',
       'Industry: type 8', 'Realtor', 'Industry: type 6'], 'Others', inplace = True)
  outlier = evaluate[evaluate['DAYS_EMPLOYED'] == 365243]
  evaluate['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True) # replace outlier by 
  evaluate['DAYS_EMPLOYED'].fillna(np.mean(evaluate['DAYS_EMPLOYED']), inplace= True) # fill na with by mean of the column
  
  # Feature engineering
  evaluate['Credit_income_ratio'] = evaluate['AMT_CREDIT']/evaluate['AMT_INCOME_TOTAL']
  evaluate['Anuity_income_ratio'] = evaluate['AMT_ANNUITY']/evaluate['AMT_INCOME_TOTAL'] 
  evaluate['Credit_term'] = evaluate['AMT_ANNUITY']/evaluate['AMT_CREDIT'] #length of the payment in months since the annuity is the monthly amount due
  evaluate['Days_employed_age'] = evaluate['DAYS_EMPLOYED']/evaluate['DAYS_BIRTH']
  
  #load modelled columns from model (173 columns sample)
  # ===
  storage_client = storage.Client()
  bucket = storage_client.get_bucket('credit-risk-bucket') 
  blob = bucket.blob('function-code/train_columns_pv1.pkl')
  blob = blob.download_as_string()
 
  byte_blob = io.BytesIO(blob)  #tranform bytes to string here

  train_ref_col = pickle.load(byte_blob)

  # Convert categorical feature to numberic using one-hot encoding
  evaluate.loc[0,'SK_ID_CURR'] = 100001
  evaluate = pd.get_dummies(evaluate, drop_first= True)
  aligned_eval, a2 = evaluate.align(train_ref_col, join='right', axis=1)
  aligned_eval # a1 adjusted imported pickle

  #Drop target if ir exists
  aligned_eval = aligned_eval.drop('TARGET', axis=1)
   
  return aligned_eval

def load_model_data(file_content):
  """ 
    Load from the bucket the trained dataset
  """
  #load modelled columns from model (173 columns sample)
  # ===
  storage_client = storage.Client()
  bucket = storage_client.get_bucket('credit-risk-bucket')
  fileContent = file_content
  blob = bucket.blob('function-code/credit_model_pv1.pkl')
  blob = blob.download_as_string()
 
  byte_blob = io.BytesIO(blob)  #tranform bytes to string here

  trained_model = pickle.load(byte_blob)
   
  return trained_model

def run_model(clean_input, trained_model):
  """
    This gets the processes the results and send data to the response function 
  """
  pred_class = trained_model.predict(clean_input)
  pred_class = pred_class[0]

  # Class Probabilities
  pred = trained_model.predict_proba(clean_input)
  pred = [p[1] for p in pred] # Positive class (1) probabilities
  pred_prob = pred[0]

  return pred_class, pred_prob