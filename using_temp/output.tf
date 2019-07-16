output "west_front_ips"{
    value="${aws_instance.west_fe.*.public_ip}"
}
output "west_be_ips"{
    value="${aws_instance.west_be.*.public_ip}"
}

output "east_front_ips"{
    value="${aws_instance.east_fe.*.public_ip}"
}
output "east_be_ips"{
    value="${aws_instance.east_be.*.public_ip}"
}

output "data_az_west"{
    value="${data.aws_availability_zones.us-west-zones.*.names}"
}