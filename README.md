This repository is for ACML2024. All experiment has implemented in C++ on a PC with Intel Xeon CPU E5-2678 v3 @ 2.50GHz and 110GB memory, running Ubuntu 22.04.

You could reproduce it with the following steps, the overall experiment may take days.


## before start, you need to:
```
apt update && \
apt install -y g++ && \
apt install -y cmake && \
apt install -y build-essential && \
apt install -y make && \
apt install wget && \
apt install zip unzip && \
apt install tar && \
apt install python3 && \
apt install python3-dev python3-pip && \
pip install numpy pandas matplotlib

```
## evaluation

Our repository contains the results in the `./output` directory and its subdirectories. You can view them directly or run the following command to see the results.

1. download datasets

```
bash ./shell/downloadDatasets.sh
```

2. build the project

```
bash ./shell/build.sh
```

3. special deal with sift2M
```
bash ./shell/specialDealWithSift2M.sh
```

4. generate index and groundTruth
```
bash ./shell/generateIndexAndGroundTruth.sh
```

5. run full_coverage scenario
```
bash ./shell/runMultWithBuild.sh
```

6. run random scenario

```
bash ./shell/runRandomWithBuild.sh
```

7. run new_insert scenario

```
bash ./shell/runNewInsertWithBuild.sh
```

8. run backup

```
bash ./shell/runBackUp.sh
```

9. Draw the result Figure
```
bash ./shell/runDrawFigure.sh
```


## Viewing Results

After drawing the result figures, you can find them in the `./output` directory and its subdirectories.




