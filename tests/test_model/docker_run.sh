mount_path=$(pwd)

docker run -v ${mount_path}:/workspace -p 8888:8888 -p 6006:6006 -it ml-container