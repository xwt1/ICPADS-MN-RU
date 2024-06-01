before start:
```
apt update 

apt-get install libmlpack-dev

apt-get install libomp-dev

apt install libeigen3-dev

```
Modify hnswlib_delete/hnswlib CMakeList.txt to avoid conflict and collision between namespace


first point: to prove some point can not be found after delete and update many times ,compile ./src/freshdiskann_prove.cpp and run. you will get something like this:

```
    Iteration 1000:
    RECALL: 0.91652
    Query Time: 0.00229733 seconds
    Delete Time: 0.132892 seconds
    Add Time: 38.103 seconds
    Query 0:
    Labels length: 999692,只能找到这么多的点
    Query 1:
    Labels length: 999692,只能找到这么多的点
    Query 2:
    Labels length: 999692,只能找到这么多的点
    Query 3:
    Labels length: 999692,只能找到这么多的点
    Query 4:
    Labels length: 999692,只能找到这么多的点
```
then after 1000 iteration , about 300 point can't be found

second point : my delete and update algorithm run fewer time than the last HNSW replaced_update method, while keep recall stable, compile 
./src/direct_delete/direct_delete_prove.cpp and ./src/direct_delete/mark_delete.cpp and run to see result.