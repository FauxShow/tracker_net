# This script will generate a similarity matching dataset from a tracking dataset
# Each track will have all crops extracted and put into a separate folder
# Each crop will be resized to short-side 256 to save space

import sys
import os

import cv2
import json


def resize_short_side(crop, short_width=128):
    h,w,_ = crop.shape
    ratio = w/h
    if h < w:
        return cv2.resize(crop, (int(short_width * ratio), short_width))
    else:
        return cv2.resize(crop, (short_width, int(short_width / ratio)))
        
    

if __name__ == "__main__":
    vid_dir = sys.argv[1]
    anno_dir = sys.argv[2]
    out_dir = sys.argv[3]

    for anno_file in os.listdir(anno_dir):
        with open(os.path.join(anno_dir, anno_file), 'r') as f:
            anno = json.load(f)
        video_path = os.path.join(vid_dir, anno_file[:-4] + 'mp4')

        video_name = anno_file[:-5]
        print(f"processing {video_name}")
        output_dirs = {} #maps the track id to the crops directory path for that track
        # create the directories for the crops
        for obj in anno['subject/objects']:
            dir_name = video_name + f"_{obj['tid']}"
            dir_path = os.path.join(out_dir, dir_name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            output_dirs[obj['tid']] = dir_path

        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if not ret:
                continue
    
            if frame_count >= len(anno['trajectories']):
                continue
            frame_anno = anno['trajectories'][frame_count]
            for item in frame_anno:
                box = item['bbox']
                track_out_dir = output_dirs[item['tid']] 
                l ,t, r, b = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                crop = frame[ t:b, l:r ]
                crop = resize_short_side(crop)
                crop_name = f"{video_name}_{item['tid']}_{frame_count}.jpg"
                cv2.imwrite(os.path.join(track_out_dir, crop_name), crop)
                cv2.waitKey(0)
            frame_count += 1

