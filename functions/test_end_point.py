# Sends an endpoint call to validate function calls
# Check invalid and missing file
# Check valid file reply
from io import TextIOWrapper
from flask import Flask, redirect, url_for, render_template, request, session
from datetime import timedelta
from werkzeug import secure_filename
#from google.cloud import storage

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

@app.route('/')
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run()