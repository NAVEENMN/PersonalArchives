output "public_ip_addresses" {
    value="${aws_instance.vm.*.public_ip}"
}

output "bubba_data" {
    value="${aws_instance.vm.*.private_ip}"
}