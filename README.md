# Download_IMDb-Face 
The code for download IMDb-Face dataset
## Download the csv file
GoogleDrive Download: https://drive.google.com/open?id=134kOnRcJgHZ2eREu8QRi99qj996Ap_ML

BaiduDrive Download: https://pan.baidu.com/s/1eRylM-jMgjYL6cyU6qQd8g
## Using csv file download images
```
python download-imdb.py
```
or
```
python download.py --input_file IMDb-Face.csv --output_dir Img --worker_num 32
```
## Get the success_file.txt
```
python get_all_list.py
```
## Using success_file.txt checks images
```
python check_jpg.py --input success_file.txt --choose 1
```
## Face detection and alignment
```
cd RetinaFace
python test_my.py
```
