# public cloud providers
provider "aws"{
  region = "${var.aws_region}"
}

# resources to deploy in this region
resource "aws_instance" "webserver"{
  ami = "${lookup(var.ami,var.aws_region)}"
  instance_type = "t2.medium"
  key_name = "nmysore"
  subnet_id = "${aws_subnet.My_VPC_Subnet.id}"
  vpc_security_group_ids = ["${aws_security_group.My_VPC_Security_Group.id}"]
}

# resources to deploy in this region
resource "aws_instance" "appserver"{
  ami = "${lookup(var.ami,var.aws_region)}"
  instance_type = "t2.medium"
  key_name = "nmysore"
  subnet_id = "${aws_subnet.My_VPC_Subnet.id}"
  vpc_security_group_ids = ["${aws_security_group.My_VPC_Security_Group.id}"]
}

output "webserver_ip_addr" {
  value       = aws_instance.webserver.public_ip
  description = "The public IP address of the main server instance."
}

output "webserver_dns" {
  value = aws_instance.webserver.public_dns
  description = "The public dns of the main server instance."
}

output "appserver_ip_addr" {
  value       = aws_instance.appserver.public_ip
  description = "The public IP address of the main server instance."
}

output "appserver_dns" {
  value = aws_instance.appserver.public_dns
  description = "The public dns of the main server instance."
}

output "webserver_login_details" {
  value = "ssh -L localhost:8888:localhost:8888 -i <pem file> ubuntu@${aws_instance.webserver.public_ip}"
}

output "appserver_login_details" {
  value = "ssh -L localhost:8888:localhost:8888 -i <pem file> ubuntu@${aws_instance.appserver.public_ip}"
}
