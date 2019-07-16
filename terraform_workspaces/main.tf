# use this s3 bucket as my backend for state management
# its good to maintain state in a sharable storage not on a local disk
# terraform automatically creates bubba on s3 and uploads the tfstate upon tf apply
# here we are using s3 as a state management not as a resource, you can use s3 as a resource also.
terraform {
    backend "s3" {
        bucket="mysterra" # your bucket name
        key="bubba/terraform.tfstate"
        region="us-west-1"
    }
}

provider "aws"{
    region = "us-east-1"
}

locals {
    vm_name="${join("-", list(terraform.workspace, "vm"))}"
}

resource "aws_instance" "myvm"{

    tags = {
        Name = "${local.vm_name}"
    }
    ami = "ami-026c8acd92718196b"
    instance_type = "t2.micro"
}