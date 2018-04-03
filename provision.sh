# Provision p2.xlarge on AWS.
docker-machine create --driver amazonec2 \
                      --amazonec2-region us-west-2 \
                      --amazonec2-zone b \
                      --amazonec2-ami ami-efd0428f \
                      --amazonec2-instance-type p2.xlarge \
                      --amazonec2-vpc-id FILL_IN_YOURS \
                      --amazonec2-access-key FILL_IN_YOURS \
                      --amazonec2-secret-key FILL_IN_YOURS \
                      --amazonec2-security-group FILL_IN_YOURS \
                      aws01

# Restart the instance first, to be sure we are running the latest installed kernel
docker-machine restart aws01

# Send our files over
docker-machine scp -r . aws01:/home/ubuntu

# Install official NVIDIA driver package
docker-machine ssh aws01 -- "sudo bash /home/ubuntu/utils/remote_setup.sh"

# Reboot to complete installation of the NVIDIA driver
docker-machine restart aws01

