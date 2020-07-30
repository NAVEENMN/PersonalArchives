# grab the information about the provider

data "aws_region" "east"{
}

data "aws_region" "west"{
provider="aws.bubba-west"
}

data "aws_availability_zones" "us-east-zones"{}

data "aws_availability_zones" "us-west-zones"{
   provider="aws.bubba-west"
}

data "aws_ami" "latest-ubuntu-east" {
most_recent = true
owners = ["099720109477"] # Canonical

  filter {
      name   = "name"
      values = ["ubuntu/images/hvm-ssd/ubuntu-xenial-16.04-amd64-server-*"]
  }

  filter {
      name   = "virtualization-type"
      values = ["hvm"]
  }
}

# aws terraform module has this feature to pull latest ami
data "aws_ami" "latest-ubuntu-west" {
provider="aws.bubba-west"
most_recent = true
owners = ["099720109477"] # Canonical

  filter {
      name   = "name"
      values = ["ubuntu/images/hvm-ssd/ubuntu-xenial-16.04-amd64-server-*"]
  }

  filter {
      name   = "virtualization-type"
      values = ["hvm"]
  }
}