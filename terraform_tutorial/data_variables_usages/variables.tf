# there is something call local variables and also global variables

variable "multi-region-deployment"{
    default=true
}

variable "env-name"{
    default="tf-demo"
}

# default type if string
variable "amis" {
    type="map"
    default={
        "us-east-1"="ami-0b898040803850657"
        "us-west-2"="ami-082b5a644766e0e6f"
    }
}