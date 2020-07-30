resource "aws_api_gateway_rest_api" "db_api_gateway" {
  name = "DBAPIGateway"
  description = "Serverless Application Gateway"
}

output "base_url" {
  value = "${aws_api_gateway_deployment.serverless.invoke_url}"
}
