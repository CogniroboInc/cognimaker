mount_path=$(pwd)

docker run -v ${mount_path}:/test_model -p 8888:8888 -p 6006:6006 -it cognimaker-test