# Visualise a video from the vidvrd dataset.
# This will draw tracked bboxes over the video

import sys
from random import randint

import cv2
import json


if __name__ == "__main__":
    vid_path = sys.argv[1]
    anno_path = sys.argv[2]
    
    with open(anno_path, 'r') as f:
        anno = json.load(f)

    colours = [ (randint(0,255), randint(0,255), randint(0,255)) for _ in range(1000) ]

    cap = cv2.VideoCapture(vid_path)

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
            colour = colours[item['tid']]
            cv2.rectangle(frame, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), colour, 1)

        print(frame_count)

        cv2.imshow('', frame)
        cv2.waitKey(30)
        frame_count += 1


