PROJECT_ID=​"kaggle-siim-healthcare"
REGION=​"us-central1"
DATASET_ID=​"siim-pneumothorax"

DICOM_STORE_ID_TEST=​"dicom-images-test"
DIRECTORY_TEST=​"test"

BUCKET="gs://pneumothorax-seg"

curl -X POST \
    -H ​"Authorization: Bearer "​$(gcloud auth print-access-token) \
    -H ​"Content-Type: application/json; charset=utf-8"​ \
    --data "{
        'gcsDestination': {
            'uriPrefix': '​"​${BUCKET}${DIRECTORY_TEST}​"​'
        }
    }" \
    "https://healthcare.googleapis.com/v1beta1/projects/​${PROJECT_ID}​/locations/​${REGION}​/datasets/​${DATASET_ID}​/dicomStores/​${DICOM_STORE_ID_TEST}​:export"

