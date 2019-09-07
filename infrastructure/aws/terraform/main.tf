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

/*
# resources to deploy in this region
resource "aws_instance" "appserver"{
  ami = "${lookup(var.ami,var.aws_region)}"
  instance_type = "t2.medium"
  key_name = "nmysore"
  subnet_id = "${aws_subnet.My_VPC_Subnet.id}"
  vpc_security_group_ids = ["${aws_security_group.My_VPC_Security_Group.id}"]
}
*/

# resources to deploy in this region
resource "aws_instance" "dbserver"{
  ami = "${lookup(var.ami,var.aws_region)}"
  instance_type = "t2.medium"
  key_name = "nmysore"
  subnet_id = "${aws_subnet.My_VPC_Subnet.id}"
  vpc_security_group_ids = ["${aws_security_group.My_VPC_Security_Group.id}"]
}

output "webserver_ip_addr" {
  value = { "ip": aws_instance.webserver.public_ip,
            "dns": aws_instance.webserver.public_dns,
            "login": "ssh -L localhost:8888:localhost:8888 -i <pem file> ubuntu@${aws_instance.webserver.public_dns}"}
  description = "The public IP address of the web server instance."
}

output "dbserver_ip_addr" {
  value = { "ip": aws_instance.dbserver.public_ip,
            "dns": aws_instance.dbserver.public_dns,
            "login": "ssh -L localhost:8888:localhost:8888 -i <pem file> ubuntu@${aws_instance.dbserver.public_dns}"}
  description = "The public IP address of the db server instance."
}

resource "local_file" "webserver_config" {
  content  = aws_instance.webserver.public_ip
  filename = "../ansible/webserver.config"
}

resource "local_file" "dbserver_config" {
  content  = aws_instance.dbserver.public_ip
  filename = "../ansible/dbserver.config"
}

/*
output "appserver_ip_addr" {
  value = { "ip": aws_instance.appserver.public_ip,
            "dns": aws_instance.appserver.public_dns,
            "login": "ssh -L localhost:8888:localhost:8888 -i <pem file> ubuntu@${aws_instance.appserver.public_dns}"}
  description = "The public IP address of the app server instance."
}
*/
