# Infrastructure

We will two tools to deploy the infrastructure we want. Terraform and ansible. Terraform is a infrastructure deployment tool which spins up a cloud infra with just few scripts. Once the infrastructure is up we will use ansible to provision the infrastructure. provisioning means installing packages or setting up code bases etc.


### Prerequisites
Since we will be deploying our infrastructure on AWS lets first configure our aws cli.
Install aws cli
```
$ pip3 install awscli --upgrade --user
```
Make sure python3 is in the environment path
```
$ tr ':' '\n' <<< "$PATH"
```
Check if you can run --version
```
$ aws --version
aws-cli/1.16.220 Python/3.7.3 Darwin/15.0.0 botocore/1.12.210
```
configure aws credentials
```
on browser: aws account >> click on your name >> My security credentials >> Access keys >>
```
create a new access key and save the details. Now run aws configure on cli and provide details.
check out which region is close to you [regions] (https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html)
 
configure aws cli
```
$ aws configure
AWS Access Key ID [****************]:
AWS Secret Access Key [****************]:
Default region name [us-west-2]: < pick a region close to you >
Default output format [json]:
```
### Installing terraform

Download appropriate package for your system.
```
https://www.terraform.io/downloads.html 
```
Where ever you have downloaded, Extract the contents and copy the application to ~/usr/local/bin. I had downloaded to ~/Downloads so I ran
```
$ cp ~/Downloads/terraform/terraform /usr/local/bin/
``` 
Now you should be able to run the command terraform --version
```
$ terraform --version
Terraform v0.12.6
```

You can also install it via brew
```
brew install terraform
```

### Installing Ansible

Install ansible via brew for mac
```
brew install ansible
```
