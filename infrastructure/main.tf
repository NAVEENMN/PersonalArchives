# public cloud providers
provider "aws"{
  region = "${var.aws_region}"
}

# resources to deploy in this region
resource "aws_instance" "myvm"{
  ami = "${lookup(var.ami,var.aws_region)}"
  instance_type = "t2.medium"
  key_name = "nmysore"
  subnet_id = "subnet-03d54fdb2d1c1b555"
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
