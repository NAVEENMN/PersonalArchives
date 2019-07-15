provider "aws"{
    region = "eu-west-1"
}

# by default it deploys in the same default availiblity zones ( data center)
# what if I want to control it ?
# set up availiblity zones.
variable "zones"{
    default = ["eu-west-1a", "eu-west-1b"]
}


resource "aws_instance" "vm"{
    count = 2
    availability_zone="${var.zones[count.index]}"
    ami = "ami-01e6a0b85de033c99"
    instance_type = "t3.micro"
}