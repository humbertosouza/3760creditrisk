# 3760 Credit Risk Assessment

## For sharing project files related to 3760 - Credit Risk Assessment

Ali Ghorbani
Humberto Ribeiro de Souza

## About the Problem

This repository contains the PPTX/PDF depicting the problem, solution and chosen architecture. Overall, a developed model was hypothetically approved by a company to go live on cloud.
The PDF file can be accessed [here.](./CreditScoreAssessmentPresentation.pdf). 

## Results

For high-availability, low cost and maintenance, Google Cloud Platform cloud provider was selected for this endeavour. The reasons for this choise are found in the PPTx file. Hence, the code uses GCP resources to use Cloud Storage Services, FireStore as Datastore, Vertex AI Notebook, IAM and Cloud Functions. 

## Usage

The POST method can be called from an authorized service account at the endpoint of the

`http://<gcplocation>/get_credt_score`

. The expected return is
'''javascript
{"status":"Aproved","default_probability":"0.213467"}
'''

## Discussion

For traceability, the transactions are saved in a NoSQL document-like database. This is believed to be useful for billing queries at low cost. In addition to it, the same structure can be used for audito purposes, for it contains non-sensitive data of the credit applicants still providing typical audit requests. If a given request cannot be provided by an existing endpoint, it can still be queried.

## TODOs

Replication and control over changes are expected to be done through change requests. The infrastructe would benefit from it when this environment is packed on a IaaS structure. Terraform can be evaluated for this purpose.
