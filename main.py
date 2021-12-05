"""
  requirements.txt
    # Function dependencies, for example:
    # package>=version
    gcloud
    oauth2client
"""


from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os



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
  clean_input = request_cleanup(request)
  if clean_input == -1:
    return '{"Error":"Incorrect or missing input parameters"}'
  trained_model = load_model_data()
  #credit_Score = run_model(clean_input, trained_model)
  return '{"score":"Reproved"}'
  """
    content_type = 'multipart/form-data; boundary=--------------------------824952619060096650875368'
    cut = content_type[0:19]  
    return cut
  """

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

