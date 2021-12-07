# 3760 Credit Risk Assessment

## For sharing project files related to 3760 - Credit Risk Assessment

Ali Ghorbani

Humberto Ribeiro de Souza

## About the Problem

This repository contains the PPTX/PDF depicting the problem, solution and chosen architecture. Overall, a developed model was hypothetically approved by a company to go live on cloud.
The PDF file can be accessed [here.](./CreditScoreAssessmentPresentation.pdf) 

A Word document close to a SOW format is also attached. It focus on the premises taken when the project started and was updated accordingly. It does not contain timelines or budget information. It contains the CLI for configuring a non-Google Linux VM. It can be dowloaded [here.](./3760-Term%20Project%20-%20Credit%20Risk%20Assessment.docx)

## Results

For high-availability, low cost and maintenance, Google Cloud Platform cloud provider was selected for this endeavour. The reasons for this choise are found in the PPTx file. Hence, the code uses GCP resources to use Cloud Storage Services, FireStore as Datastore, Vertex AI Notebook, IAM and Cloud Functions. 

## Usage

The POST method can be called from an authorized service account at the endpoint of the

`http://<gcplocation>/get_credt_score`

A form-data must be sent in the body having as parameters

record: "single-record-file.csv"
logged_user: SSO or logged user from the source system. E.g. johnsmith@outlook.com

The record field is mandatory. The expected file format can be found on [funcitons/record_eval1.csv](./functions/record_eval1.csv)

Auth Data must have a bearer token. This endpoint is intended to query an Google Cloud Platform endpoint and expects a service account to be set up and authorized for the services it requires. The standard utilized in this project follows svc_servicename standard.

If querying via curl, use  

```bash
<pre>curl -L  -H &quot;Authorization: bearer $(gcloud auth print-identity-token)&quot; --data-binary @record_req1.csv &quot;https://<server_address>/get_credit_score&quot;</pre>
```

If running it locally, it will trigger Flask and server it at 127.0.0.1:5000. Please have in mind that gcloud sdk and the environment variables must be set for this to work.


### Return

The expected return is the status analysis and the default probability in a JSON format as follows.

```javascript
{"status":"Approved","default_probability":"0.213467"}
```

## Discussion

For traceability, the transactions are saved in a NoSQL document-like database. This is believed to be useful for billing queries at low cost. In addition to it, the same structure can be used for audit purposes, for it contains non-sensitive data of the credit applicants still providing typical audit requests. If a given request cannot be provided by an existing endpoint, it can still be queried.

## TODOs

Replication and control over changes are expected to be done through change requests. The infrastructe would benefit from it when this environment is packed on a IaaS structure. Terraform can be evaluated for this purpose.
