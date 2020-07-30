# eu-east-1
# eu-west-1
provider "aws"{
    region = "us-east-1"
}

# depends_on : make forend unavailable untill both backend is not available
# life_cycle : lets a new change needs to applied to an instance,
# instead of bringing it down deploy a new instance and then bring this down. default if opposite
resource "aws_instance" "frontend"{
    ami = "ami-026c8acd92718196b"
    instance_type = "t2.micro"
    depends_on = ["aws_instance.backend"]
    lifecycle {
        create_before_destroy = true
    }
}

# timeout: by default they have rest command timeout, you can set your own timeout
# ( this is behaviour change not change to instance itself )
resource "aws_instance" "backend"{
    count = 2
    ami = "ami-026c8acd92718196b"
    instance_type = "t2.micro"
    timeouts {
        create="60m"
        delete="2h"
    }
}