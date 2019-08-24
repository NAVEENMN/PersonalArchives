# public cloud providers
provider "aws"{
  region = "${var.aws_region}"
}

# create the VPC
resource "aws_vpc" "My_VPC" {
  cidr_block           = "${var.vpcCIDRblock}"
  instance_tenancy     = "${var.instanceTenancy}"
  enable_dns_support   = "${var.dnsSupport}"
  enable_dns_hostnames = "${var.dnsHostNames}"
  assign_generated_ipv6_cidr_block = true
} # end resource

# Create the Internet Gateway
resource "aws_internet_gateway" "My_VPC_GW" {
  vpc_id = "${aws_vpc.My_VPC.id}"
} # end resource

# Create the Route Table
resource "aws_route_table" "My_VPC_route_table" {
  vpc_id = "${aws_vpc.My_VPC.id}"
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = "${aws_internet_gateway.My_VPC_GW.id}"
  }

  route { 
    ipv6_cidr_block = "::/0"
    gateway_id = "${aws_internet_gateway.My_VPC_GW.id}"
  }

} # end resource

# Create the Internet Access
resource "aws_route" "My_VPC_internet_access" {
  route_table_id        = "${aws_route_table.My_VPC_route_table.id}"
  destination_cidr_block = "${var.destinationCIDRblock}"
  gateway_id             = "${aws_internet_gateway.My_VPC_GW.id}"
} # end resource

# create the Subnet
resource "aws_subnet" "My_VPC_Subnet" {
  vpc_id                  = "${aws_vpc.My_VPC.id}"
  # cidr_block              = "${var.subnetCIDRblock}"
  cidr_block              = "${cidrsubnet(aws_vpc.My_VPC.cidr_block, 4, 1)}"
  map_public_ip_on_launch = "${var.mapPublicIP}"
  availability_zone       = "${var.availabilityZone}"
  ipv6_cidr_block = "${cidrsubnet(aws_vpc.My_VPC.ipv6_cidr_block, 8, 1)}"
  assign_ipv6_address_on_creation = "${var.assignipv6AddressOnCreation}"
} # end resource

# Associate the Route Table with the Subnet
resource "aws_route_table_association" "My_VPC_association" {
    subnet_id      = "${aws_subnet.My_VPC_Subnet.id}"
    route_table_id = "${aws_route_table.My_VPC_route_table.id}"
} # end resource

# Create the Security Group
resource "aws_security_group" "My_VPC_Security_Group" {
  vpc_id       = "${aws_vpc.My_VPC.id}"
  name         = "My VPC Security Group"
  description  = "My VPC Security Group"
  ingress {
    cidr_blocks = "${var.ingressCIDRblock}"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
  }
  ingress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    ipv6_cidr_blocks = ["::/0"]
  }
  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    ipv6_cidr_blocks = ["::/0"]
  }
} # end resource

# resources to deploy in this region
resource "aws_instance" "myvm"{
  ami = "${lookup(var.ami,var.aws_region)}"
  instance_type = "t2.medium"
  key_name = "nmysore"
  subnet_id = "${aws_subnet.My_VPC_Subnet.id}"
  vpc_security_group_ids = ["${aws_security_group.My_VPC_Security_Group.id}"]
}

output "instance_ip_addr" {
  value       = aws_instance.myvm.public_ip
  description = "The public IP address of the main server instance."
}

output "instance_dns" {
  value = aws_instance.myvm.public_dns
  description = "The public dns of the main server instance."
}

output "login_details" {
  value = "ssh -L localhost:8888:localhost:8888 -i  ubuntu@${aws_instance.myvm.public_ip}"
}
