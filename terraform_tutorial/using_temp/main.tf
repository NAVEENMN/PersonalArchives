provider "aws"  {
    region = "us-east-1"
}

provider "aws" {
    alias="bubba-west"
    region = "us-west-2"
}


locals {
    def_front_name="${join("-",list(var.env-name,"frontend"))}"
    def_back_name="${join("-",list(var.env-name,"backend"))}"
}

resource "aws_instance" "west_fe"{
    tags {
        Name="${local.def_front_name}"
    }
    count=2
    depends_on = ["aws_instance.west_be"]
    availability_zone ="${data.aws_availability_zones.us-west-zones.names[count.index]}"
    provider="aws.bubba-west"
    ami="${data.aws_ami.latest-ubuntu-west.id}" #"${var.amis[data.aws_region.west.name]}"
    instance_type="t2.micro"
}

resource "aws_instance" "west_be"{
    
     tags {
        Name="${local.def_back_name}"
    }
    count=2
    availability_zone ="${data.aws_availability_zones.us-west-zones.names[count.index]}"
    provider="aws.bubba-west"
    ami="${data.aws_ami.latest-ubuntu-west.id}"
    instance_type="t2.micro"
}

resource "aws_instance" "east_fe"{
    tags {
        Name="${local.def_front_name}"
    }
    count="${var.multi-region-deployment ? 1:0}"
    depends_on = ["aws_instance.east_be"]
    availability_zone ="${data.aws_availability_zones.us-east-zones.names[count.index]}"
    ami="${data.aws_ami.latest-ubuntu-east.id}"
    instance_type="t2.micro"
}

resource "aws_instance" "east_be"{
    
     tags {
        Name="${local.def_back_name}"
    }
    count="${var.multi-region-deployment ? 1:0}"
    availability_zone ="${data.aws_availability_zones.us-east-zones.names[count.index]}"
    ami="${data.aws_ami.latest-ubuntu-east.id}"
    instance_type="t2.micro"
}
