# Regions Name            |  Region
# US West (N. California) | us-west-1
# US West (Oregon)        | us-west-2
# Asia Pacific (Singapore)| ap-southeast-1

variable "aws_region" {
  default = "us-west-2"
}

# Deep Learning AMI (Ubuntu) Version 24.0
variable "ami" {
  type = "map"
  default = {
    "us-west-1" = "ami-0a9f680d70a2fc2a9"
    "us-west-2" = "ami-0ddba16a97b1dcda5"
    "ap-southeast-1" = "ami-0c2f3e32c99bdf1cf"
  }
}
