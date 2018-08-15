#!/bin/bash
root_path="$1"
video1="$2"
video2="$3"
model_name="$4"

path_dataset="$root_path/datasets"
cd "$path_dataset"
path_model="$path_dataset/$model_name"
mkdir "$path_model"
cd "$path_model"

path_video1="$path_model/testA"
path_video2="$path_model/testB"
mkdir "$path_video1"
mkdir "$path_video2"

cd "$path_video1" #-r means fps 5 is 5 pics per sec
ffmpeg -i "$root_path/videos/$video1" -ss 3 -t 140 -r 29 -f image2 image-%4d.jpeg &> /dev/null
echo "video 1 is done!"


cd "$path_video2"
ffmpeg -i "$root_path/videos/$video2" -ss 16 -t 50 -r 29 -f image2 image-%4d.jpeg &> /dev/null
echo "video 2 is done!"
exit 0
# 10 - 20
