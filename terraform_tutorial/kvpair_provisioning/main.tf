# generate key pair on aws and download it and 
# chmod 400 *.pem file
terraform {
    backend "s3" {
        bucket="mysterra" # your bucket name
        key="bubba/terraform.tfstate"
        region="us-west-1"
    }
}

provider "aws"{
    region = "us-west-1"
}

locals {
    vm_name="${join("-", list(terraform.workspace, "vm"))}"
}


# adding a provisioner called remote-exe
# they are many different types of provisioner
# one of the provisioner called file provisioner will upload a file from 
# local directory to vm 
# cloud init is used on the aws side which does this file provisioning 
# they is also a provisioner for ansible 
resource "aws_instance" "myvm"{

    tags = {
        Name = "${local.vm_name}"
    }
    ami = "ami-056ee704806822732"
    instance_type = "t1.micro"
    key_name="mysterra_kv"
    security_groups = ["default"] # make sure on aws under security group, ssh is made inbound for default
    provisioner "remote-exec"{
        inline=["touch test.dat"] # create a test.dat file on vm 
        connection {
            host = "${aws_instance.myvm.public_ip}" 
            type = "ssh"
            user = "ubuntu" # user name comes from ami creator
            private_key = "${file("/home/ec2-user/environment/mysterra_kv.pem")}"
        }
    }
}