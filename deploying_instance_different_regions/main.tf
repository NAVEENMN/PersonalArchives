# first one will be the default provider
provider "aws" {
    region = "eu-west-1"
}

# you can create alias instead of remembering sa-east-1
# you can change the region while alias remains the same.
provider "aws" {
    alias="europe-west"
    region = "sa-east-1"
}

# westvm is just name to refer
resource "aws_instance" "westvm1" {
    ami = "ami-01e6a0b85de033c99"
    instance_type="t3.micro"
}

# when we have muliple provider regions
# if not mentioned by default it picks first one
resource "aws_instance" "westvm2" {
    provider="aws.europe-west"
    ami = "ami-0dcf15c37a41c965f"
    instance_type="t3.micro"
}