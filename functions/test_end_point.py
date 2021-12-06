# For testing local functions and
# Sends an endpoint call to validate function calls
# Check invalid and missing file
# Check valid file reply

import io
from flask import Flask, redirect, url_for, render_template, request, session
from datetime import timedelta
from werkzeug import secure_filename
from google.cloud import storage
import os
import sys

import pickle
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
# Ignore ConvergenceWarning messages
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


app = Flask(__name__)
app.secret_key = "27eduCBA09"

from flask import request, jsonify

@app.route('/test', methods = ['POST']) 
def test_all():
    user_data = request.get_json() # 
    record = request.form.get('record')
    # add here the code to create the user
    res = '{‘status’: ‘ok’}'
    print ('request.data: '+ str(request.data))
    print ('request.form: '+ str(request.form))
    print ('request.files: '+ str(request.files['record']))
    print ('request.values: '+ str(request.values))
    print ('request.json: '+ str(request.data))
    fileContent = request.files['record']
    fileContent.save(secure_filename('new_'+fileContent.filename))
    text_content = ''
    with open('new_'+fileContent.filename, 'r') as f:
        text_content= f.read()

    print('file content: '+ text_content)
    return jsonify(res)

@app.route('/storage', methods = ['POST']) 
def test_storage():
  

  credential_path = "/home/humberto/gcloud/svc_credituser_key.json"
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path  

  storage_client = storage.Client()
  bucket = storage_client.get_bucket('credit-risk-bucket')
  fileContent = request.files['record']
  blob = bucket.blob(secure_filename('function_code_new_'+fileContent.filename))
  file_content = request.files['record'].read()
  blob.upload_from_string(file_content.decode('utf-8'))

  #b = bytes(mystring, 'utf-8')
  clean_input = request_cleanup(file_content.decode('utf-8'))    

  trained_model = load_model_data()

  credit_score = run_model(clean_input, trained_model)

  result = "Approved" if credit_score[0] == 0 else "Reproved"

  print('{"score":"'+result+'","probability":"'+str(credit_score[1])+'"}')

  return '{"score":"'+result+'","probability":"'+str(credit_score[1])+'"}'

@app.route('/')
def hello():
    return "Hello World!"

def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))    


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
  fileContent = request.files['record']
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

def load_model_data():
  """ 
    Load from the bucket the trained dataset
  """
  #load modelled columns from model (173 columns sample)
  # ===
  storage_client = storage.Client()
  bucket = storage_client.get_bucket('credit-risk-bucket')
  fileContent = request.files['record']
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


if __name__ == '__main__':
    app.run()

