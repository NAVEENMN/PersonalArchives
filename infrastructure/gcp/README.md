### Setting up Infrastructre and Provisioning on Google Cloud

Setting up infrastructure is easier via ansible. But inorder to setup and provision we need to setup few things on our local system

# Google Cloud Platform Project
1. You will need to create a Google Cloud Platform Project as a first step.
2. set up or enable billing for this project.
3. In order for ansible to create Compute Engine instances we will need a google service account. [Service Account](https://cloud.google.com/compute/docs/access/service-accounts#serviceaccount) 
4. create a new JSON formatted private key file for this Service Account and download and save it, preferebly in ~/.ssh, this will be used by ansible.
5. Next you will want to install the [Cloud SDK](https://cloud.google.com/sdk/)

# Install dependencies

Basic installs
```
sudo apt-get install -y build-essential git python-dev python-pip
```
Install GCP module dependencies
```
pip install googleauth requests

## Run the playbook

Playbook create_google_compute_instance.yml will deploy the instance and destroy_google_compute_instance.yml will destroy the instances.
```
ansible-playbook -i host playbooks/create_gcp_compute_instance.yml
```
