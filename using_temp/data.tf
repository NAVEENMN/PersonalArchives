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