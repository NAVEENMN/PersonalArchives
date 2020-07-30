# Setting up Infrastructre and Provisioning on Google Cloud

Setting up infrastructure is easier via ansible. But inorder to setup and provision we need to setup few things on our local system

# Google Cloud Platform Project
1. You will need to create a Google Cloud Platform Project as a first step.
2. set up or enable billing for this project.
3. In order for ansible to create Compute Engine instances we will need a google service account. [Service Account](https://cloud.google.com/compute/docs/access/service-accounts#serviceaccount)
4. Service accounts need right admin roles to enable ssh without gcloud or for ansible to configure. While creating service cloud add these admin roles
```
roles/compute.admin
roles/iam.serviceAccountUser
roles/compute.instanceAdmin
roles/compute.instanceAdmin.v1
roles/compute.osAdminLogin
```
5. create a new JSON formatted private key file for this Service Account and download and save it, preferebly in ~/.ssh, this will be used by ansible.
6. Next you will want to install the [Cloud SDK](https://cloud.google.com/sdk/)

# Install dependencies

Basic installs
```
sudo apt-get install -y build-essential git python-dev python-pip
```
Install GCP module dependencies
```
pip install googleauth requests
```
## Run the playbook

Playbook create_google_compute_instance.yml will deploy the instance and destroy_google_compute_instance.yml will destroy the instances.
```
ansible-playbook -i host playbooks/create_gcp_compute_instance.yml
```
In order to get the instance ip run
```
gcloud compute addresses list
```
You can ssh to the instance by running
```
gcloud compute ssh <instance_name>
```
To destroy the resources
```
ansible-playbook -i host playbooks/destroy_gcp_compute_instance.yml
```
