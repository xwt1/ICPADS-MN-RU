before start:
```
apt update 

apt-get install libomp-dev

apt install libeigen3-dev
```
in main directory, execute:
```
mkdir build && cd build
cmake ..
make -j16
cd src
```
and you will find 

- compare_rawMMR_with_Dhnsw_in_ip 是内积比较程序，compare_rawMMR_With_Dhnsw是距离比较程序，generate_index是以距离作为比较函数生成索引的程序，generate_index_ip是是以内积作为比较函数生成索引的程序