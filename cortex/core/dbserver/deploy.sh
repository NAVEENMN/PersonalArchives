#aws s3api create-bucket --bucket=cortex-db-serverless --region=us-east-1
if [ -d "${PWD}/apis/Image" ]
opwd=${PWD}
then
  cd ${PWD}/apis/Image
  pip install --target . pymongo
  pip install --target . dnspython
  zip -r9 ../package.zip .
  echo ${opwd}
  cd ${opwd}/apis
  echo "uploading package to s3:"
  aws s3 cp package.zip s3://cortex-db-serverless/v1.0.0/package.zip
fi
