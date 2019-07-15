# this will be default
provider "aws"{
    region = "us-east-1"
}

provider "aws"{
    alias="bubba-west"
    region = "us-west-1"
}

# data is dynamic
# variable is static ( example : ami, it doesnt change)
data "aws_availability_zones" "us_east_zones" {
}

# you have to use alias here
data "aws_availability_zones" "us_west_zones" {
    provider="aws.bubba-west"
}

# locals ( its a keyword ) is a local block, these variables can be used locally
locals {
    def_front_name="${join("-", list(var.env-name, "frontend"))}"
    def_back_name="${join("-", list(var.env-name, "backend"))}"
}

resource "aws_instance" "west_fe" {
    tags = {
        Name="${local.def_front_name}"
    }
    count = 2
    depends_on = ["aws_instance.west_be"]
    availability_zone = "${data.aws_availability_zones.us_west_zones.names[count.index]}"
    provider="aws.bubba-west"
    ami="${var.west_ami}"
    instance_type="t2.micro"
}

# if count = 0 -> nothing to deployment
# if var.multi-region-deployment is set to true then deploy instances
# count = "${var.multi-region-deployment ? 1:0}"

resource "aws_instance" "west_be" {
    tags = {
        Name="${local.def_front_name}"
    }
    count = 2
    availability_zone = "${data.aws_availability_zones.us_west_zones.names[count.index]}"
    provider="aws.bubba-west"
    ami="${var.amis[aws.region]}"
    instance_type="t2.micro"
}
