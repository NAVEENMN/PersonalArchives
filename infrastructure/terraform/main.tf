# public cloud providers
provider "aws"{
  region = "${var.aws_region}"
}

# security groups
resource "aws_security_group" "instance" {
  name = "aws_sec_grp"
  ingress {
    from_port = 8080
    to_port = 8080
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port = 80
    to_port = 80
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# resources to deploy in this region
resource "aws_instance" "myvm"{
  ami = "${lookup(var.ami,var.aws_region)}"
  instance_type = "t2.medium"
  key_name = "nmysore"
  vpc_security_group_ids = ["${aws_security_group.instance.id}"]
}

output "instance_ip_addr" {
  value       = aws_instance.myvm.public_ip
  description = "The public IP address of the main server instance."
}

output "instance_dns" {
  value = aws_instance.myvm.public_dns
  description = "The public dns of the main server instance."
}
