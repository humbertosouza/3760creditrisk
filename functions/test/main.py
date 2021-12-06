from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os
from werkzeug.utils import secure_filename


def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        return f'Hello World!'

def process_request(request):
  """
    This will load the request having parameters. Call all the helper functions and returns the 
    credit score
  """
  text_content = ''
  credential_path = "/home/humberto/gcloud/svc_credituser_key.json"
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path  
 
   
  try:
    # get file from the endpoint and saves on bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('credit-risk-bucket')
    fileContent = request.files['record']
    blob = bucket.blob(secure_filename('function_code_new_'+fileContent.filename))
    file_content = request.files['record'].read()
    blob.upload_from_string(file_content.decode('utf-8'))

  # Catch exceptions  
  except Exception as e:
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    return '{"error":"'+ message +'"}'

  clean_input = request_cleanup(request)
  if clean_input == -1:
    return '{"Error":"Incorrect or missing input parameters"}'
  trained_model = load_model_data()
  #credit_Score = run_model(clean_input, trained_model)
  return '{"score":"Reproved"}'


def request_cleanup(request):
  """ 
    Clean up the input data
  """
  return ''

def load_model_data():
  """ 
    Load from the bucket the trained dataset
  """
  trained_model = ''
  return trained_model

def run_model(clean_input, trained_model):
  """
    This gets the processes the results and send data to the response function 
  """
  return ''

print (process_request(''))
