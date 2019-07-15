# to run terraform apply -var-file variables.tfvars
# if you variables files in name terraform.tfvars then it will be automatically picked up

provider "aws"{
    region = "eu-west-1"
}

resource "aws_instance" "vm" {
    count = 2
    availability_zone="${var.zones[count.index]}"
    ami = "ami-01e6a0b85de033c99"
    instance_type = "t3.micro"
}