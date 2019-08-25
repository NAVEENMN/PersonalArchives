# Regions Name            |  Region
# US West (N. California) | us-west-1
# US West (Oregon)        | us-west-2
# Asia Pacific (Singapore)| ap-southeast-1

variable "aws_region" {
  default = "us-west-2"
}

# Deep Learning AMI (Ubuntu) Version 24.0
variable "dl_ami" {
  type = "map"
  default = {
    "us-west-1" = "ami-0a9f680d70a2fc2a9"
    "us-west-2" = "ami-0ddba16a97b1dcda5"
    "ap-southeast-1" = "ami-0c2f3e32c99bdf1cf"
  }
}

# Ubuntu Server 18.04 LTS - free tier
variable "ami" {
  type = "map"
  default = {
    "us-west-1" = "ami-08fd8ae3806f09a08"
    "us-west-2" = "ami-06f2f779464715dc5"
    "ap-southeast-1" = "ami-03b6f27628a4569c8"
  }
}

variable "availabilityZone" {
 default = "us-west-2a"
}
variable "instanceTenancy" {
 default = "default"
}
variable "dnsSupport" {
 default = true
}
variable "dnsHostNames" {
        default = true
}
variable "vpcCIDRblock" {
 default = "10.0.0.0/16"
}
variable "assignipv6AddressOnCreation" {
 default = true
}

variable "subnetCIDRblock" {
        default = "10.0.1.0/24"
}
variable "destinationCIDRblock" {
        default = "0.0.0.0/0"
}
variable "ingressCIDRblock" {
        type = "list"
        default = [ "0.0.0.0/0" ]
}
variable "mapPublicIP" {
        default = true
}
