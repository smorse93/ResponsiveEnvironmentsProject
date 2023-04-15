#annotator stuff
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

#general
from operator import index
import cv2
import argparse
from pyparsing import results
#from torch import true_divide
from ultralytics import YOLO
import supervision as sv
import numpy as np
import math
import torch

import pygame

import sys
sys.path.insert(0, './Ableton/User Library/Remote Scripts/AbletonOSC')
import AbletonTest

# initialize Pygame
import pygame

# initialize Pygame
pygame.init()

# set up screen dimensions
screen_width = 1080
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))

# set up circle dimensions
circle_radius = 25
circle_color = (255, 0, 0)
circle_x = screen_width // 2
circle_y = screen_height // 2



colors = sv.ColorPalette.default()

zoneDefs = np.array([ 640, 1080, 360, 720])

#create polygons
polygons = [
    np.array([
        [0, 0],
        [640 , 0],
        [640, 360],
        [0, 360 ]
    ], np.int32), 
    np.array([
        [640 , 0],
        [1280, 0],
        [1280, 360 ],
        [640 ,360]
    ], np.int32), 
    np.array([
        [0, 360],
        [640 , 360],
        [640, 720],
        [0, 720 ]
    ], np.int32), 
    np.array([
        [640 , 360],
        [1280, 360],
        [1280, 720 ],
        [640, 720]
    ], np.int32), 
   
]



#setting up webcam and resolution
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args






#area of bounding box
def area (p1_loc, p2_loc):
    area_p1 = np.array(abs(p1_loc[2] - p1_loc[0])*abs(p1_loc[3] - p1_loc[1]))
    area_p2 = np.array(abs(p2_loc[2] - p2_loc[0])*abs(p2_loc[3] - p2_loc[1]))
    return area_p1, area_p2

#Change in bounding box area
def areaChange (area_p1, area_p1_prev):
    area_change_p1 = abs(area_p1 - area_p1_prev)
   
    #normalize the area change
    area_change_p1 = area_change_p1/area_p1
    

    return area_change_p1

#centroid finding
def centroid (p1_loc, p2_loc):
    diff_xy_p1 = np.array([p1_loc[2] - p1_loc[0],p1_loc[3] - p1_loc[1]])
    diff_xy_p2 = np.array([p2_loc[2] - p2_loc[0],p2_loc[3] - p2_loc[1]])
    print('diff_xy')
    print(diff_xy_p1)
    print(diff_xy_p2)
    centroid_p1 = np.array([(p1_loc[0]+(diff_xy_p1[0]/2)), (p1_loc[1]+(diff_xy_p1[1]/2))])
    centroid_p2 = np.array([(p2_loc[0]+ (diff_xy_p2[0]/2)), (p2_loc[1]+(diff_xy_p2[1]/2))])
    return centroid_p1, centroid_p2

#distance
def distance (centroid_p1, centroid_p2, p2_detect):
    if p2_detect == True:
        dist = math.sqrt( ((centroid_p1[0]-centroid_p2[0])**2) + ((centroid_p1[1]-centroid_p2[1])**2))
    else:
        dist = 1000
    return dist

#motion
def motion (centroid_p1, centroid_p1_prev, centroid_p2, centroid_p2_prev):
    #get distance change between frames
    motionp1 = math.sqrt( ((centroid_p1[0]-centroid_p1_prev[0])**2) + ((centroid_p1[1]-centroid_p1_prev[1])**2))
    motionp2 = math.sqrt( ((centroid_p2[0]-centroid_p2_prev[0])**2) + ((centroid_p2[1]-centroid_p2_prev[1])**2))
    
    #normalize to a useable range
    normMotionp1 = (motionp1/500)*10
    normMotionp2 = (motionp2/500)*10
    return normMotionp1, normMotionp2

#centroid zones
def zonesDetect (centroid_p):
    
    #if centroid x coordinate is greater than the x coordinate of the zone...
    if centroid_p[0] > zoneDefs[0]:
        #we now know it is on the right half
        if centroid_p[1] > zoneDefs[2]:
            #we now know it is on the bottom half
            zone_p = 3
        else:
            #we now know in the top half
            zone_p = 1
    else:
        #we now know in the left half split down the middle vertically
        if centroid_p[1] > zoneDefs[2]:
            #we now know in the bottom half
            zone_p = 2
        else:
            #we now know in the top half
            zone_p = 0

    return zone_p

#--------------------- MAIN ----------------------
def main():
    #getting webcam to run
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    #for uploading a video (you will need to comment out the cap stuff below)
    #cap = cv2.VideoCapture('testClipStevenAnaQuick.mp4')

    #for live feed video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    #set model 
    model = YOLO("yolov8l.pt")

    #create multiple zones under zones
    zones = [
        sv.PolygonZone(
            polygon=polygon, 
            frame_resolution_wh=tuple(args.webcam_resolution)
        )
        for polygon
        in polygons
    ]
    #annotate zones
    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone, 
            color=colors.by_idx(index), 
            thickness=4,
            text_thickness=8,
            text_scale=4
        )
        for index, zone
        in enumerate(zones)
    ]
    
    #create annotation identities for each item in the index
    box_annotators = [
    sv.BoxAnnotator(
        color=colors.by_idx(index), 
        thickness=4, 
        text_thickness=4, 
        text_scale=2
        )
    for index
    in range(len(polygons))
    ]
    
    #variable definition
    p1_loc = np.array([0, 0, 0, 0])
    p2_loc = np.array([0, 0, 0, 0])
    centroid_p1 = None
    centroid_p2 = None
    centroid_p1_prev = None
    centroid_p2_prev = None
    area_p1 = None
    area_p2 = None
    area_p1_prev = None
    area_p2_prev = None
    area_p1_prev_prev = None
    area_p2_prev_prev = None
    area_change_p1 = None
    area_change_p2 = None
    area_change_sum = 0
    motion_p1 = 0.0
    motion_p2 = 0.0
    p1_zone = None
    prevTrackZone = None
    frameCount = 0
    listAreaChange = [0.000,0.000,0.000,0.000]


    ########################################################
    ########################################################
    ########################################################
    
    # START ABLETON TRACK
    #ex --> command_str = "/live/song/set/tempo 123.0"
    #ex --> command_str = "/live/song/set/tempo 124.0"
    
    AbletonTest.doSomething("/live/song/start_playing") #start song
    # tempo = client.query("/live/song/get/tempo")
    #print("Got song tempo: %.1f" % tempo[0])
    currentTrack = 0
    bpm = (AbletonTest.getTempo("/live/song/get/tempo_bpm"))
    volume = (AbletonTest.getVolume("/live/track/get/volume " + str(currentTrack)))
    originalBPM = bpm

    #intiate tracks volumes
    AbletonTest.doSomething("/live/track/set/volume " + str(0) + " .01")
    AbletonTest.doSomething("/live/track/set/volume " + str(1) + " .01")
    AbletonTest.doSomething("/live/track/set/volume " + str(2) + " .01")
    AbletonTest.doSomething("/live/track/set/volume " + str(3) + " .01")


    print("BPM: ",bpm)
    print("Volume: ",volume)
    bpmLowerLimit = 75
    #bpmUpperLimit = bpm + 30.0
    
    ########################################################
    ########################################################
    ########################################################

    
    while True:
        #read the current frame
        ret, frame = cap.read()

        #MPS graphics card line 
        #result = model(frame, agnostic_nms=True, device='mps')[0]
        result = model(frame, device="mps")[0]

        #put frame into model - uncomment this for intel
        #result = model(frame, agnostic_nms=True)[0]

        #get detections using the model
        detections = sv.Detections.from_yolov8(result)
       
        #limit detections to mask area, class id people, and confidence >0.5 
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
        
        #reset p1 and p2
        p1_detect = False
        p2_detect = False
        
        #set values for p1_loc and p2_loc
        if detections.xyxy.any():
            p1_loc = detections.xyxy[0]
            p1_detect = True
            
            if len(detections.xyxy) >= 2:
                p2_loc = detections.xyxy[1]      
                p2_detect = True
        
        print(f'detections{detections.xyxy}')
        print(f'p1_detect{p1_detect}')   
        print(f'p2_detect{p2_detect}')
        print(f'p1_loc{p1_loc}')
        print(f'p2_loc{p2_loc}')

        #capture prior position of people 
        if (centroid_p1 is not None):
            centroid_p1_prev = centroid_p1
            centroid_p2_prev = centroid_p2

        #getcurrent centroids
        centroid_p1, centroid_p2 = centroid(p1_loc,p2_loc)

        #get the distance between people
        dist = distance(p1_loc,p2_loc, p2_detect)

        #get the distance covered by a single person between two frames (speed)
        if (centroid_p1_prev is not None):
                motion_p1, motion_p2 = motion(centroid_p1, centroid_p1_prev, centroid_p2, centroid_p2_prev)
                print(f'motion_p1: {motion_p1}')
                print(f'motion_p2: {motion_p2}')
                
        print(f'distance:{dist}')
        print(f'centroid_p1{centroid_p1}')
        print(f'centroid_p2{centroid_p2}')

        #capture previous previous frame area 
        if (area_p1_prev is not None):
            area_p1_prev_prev = area_p1_prev
            area_p2_prev_prev = area_p2_prev

        #capture previous frame area 
        if (area_p1 is not None):
            area_p1_prev = area_p1
            area_p2_prev = area_p2
        
        #get area values
        area_p1, area_p2 = area (p1_loc, p2_loc)
               
        #get area change values
        if (area_p1_prev is not None):
            area_change_p1 = areaChange(area_p1, area_p1_prev)
            area_change_p2 = areaChange(area_p2, area_p2_prev)
            if math.isnan(area_change_p1):
                area_change_p1 = 0;
            if math.isnan(area_change_p2):
                area_change_p2 = 0;
            
            area_change_sum = area_change_p1 + area_change_p2 
        print(f'area_p1: {area_p1}')
        print(f'area_p2: {area_p2}')
        print(f'area_p1_change: {area_change_p1}')
        print(f'area_p2_change: {area_change_p2}')
        print(f'area_change_sum: {area_change_sum}')

        #dancing vs not dancing metric will need to be calculated here
            #take past three area changes and do average the change in area over the time
            #get a scale from 1 to 3 value or somethign of this - not super clear yet

        #Annotations for boxes - We don't actually need this except for the visual component
        for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = zone_annotator.annotate(scene=frame)

        #create a dict for all of our labels
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]  
        
        
        ###################################
        ###################################
        ###################################
        #Ableton testing
        #setting prev track zone if it hasn't been set yet 
        if prevTrackZone is None:
            if p1_zone is None:
                prevTrackZone = 0  
                
        #1. proximity -> volume 
        if (p1_detect and p2_detect):
            # volume ranges from 0.0 (minimum) - to 1.0 (maxiumum)
            if (dist < 200):
                AbletonTest.doSomething("/live/track/set/volume " + str(currentTrack) + " .75")
            elif (dist < 400):
                AbletonTest.doSomething("/live/track/set/volume " + str(currentTrack) + " .50")
            elif (dist < 600):
                AbletonTest.doSomething("/live/track/set/volume " + str(currentTrack) + " .25")
            elif (dist < 800):
                AbletonTest.doSomething("/live/track/set/volume " + str(currentTrack) + " .15")
            elif (dist >= 800):
                AbletonTest.doSomething("/live/track/set/volume " + str(currentTrack) + " .01")
        #commenting this out because it sometimes randomly turns the volume back up when it misses recognizing p2 for a single frame which sometimes happens
        # elif ((p1_detect and not p2_detect) or (not p1_detect and p2_detect)):
        #     AbletonTest.doSomething("/live/track/set/volume " + str(currentTrack) + " .75")
        # else:
        #     AbletonTest.doSomething("/live/track/set/volume " + str(currentTrack) + " .75")          
            
        #2. collective movement -> BPM 
        #I tried changing this up, we will see if this works better 

        #add area_change_sum to listAreaChange at position frameCount
        listAreaChange[frameCount] = area_change_sum
        print(f'listAreaChange: {listAreaChange}')
        print(f'bpm: {bpm}')
        frameCount += 1
        if frameCount == 4:
            frameCount = 0
            #reset listAreaChange to 0's
            listAreaChange = [0.000,0.000,0.000,0.000]
            #insert areaChangeSum into listAreaChange
            listAreaChange[frameCount] = area_change_sum

            #get the mean of the listAreaChange
            sumAreaChange = sum(listAreaChange)
            print(f'sumAreaChange: {sumAreaChange}')

            #if the sumAreaChange is less than 0.1, decrease the bpm by 5
            if (sumAreaChange < 0.06 and bpm > bpmLowerLimit):
                bpm -= 5.0
                AbletonTest.doSomething("/live/song/set/tempo " + str(bpm))
            elif (sumAreaChange < 0.1 and bpm > bpmLowerLimit):
                bpm -= 3.0
                AbletonTest.doSomething("/live/song/set/tempo " + str(bpm))
            #if the sumAreaChange is greater than 0.1 and bpm is not equal to originalBPM, increase the bpm by 5
            elif (sumAreaChange > 0.1 and bpm != originalBPM):
                bpm += 3.0
                if (bpm > 120):
                    bpm = 120
                AbletonTest.doSomething("/live/song/set/tempo " + str(bpm))
            elif (sumAreaChange > 0.15 and bpm != originalBPM):
                 bpm += 5.0
                 if (bpm > 120):
                     bpm = 120
                 AbletonTest.doSomething("/live/song/set/tempo " + str(bpm))
            
         

        # if ((motion_p1 + motion_p2) < .04 and bpm > bpmLowerLimit):
        #     bpm -= 10.0
        #     AbletonTest.doSomething("/live/song/set/tempo " + str(bpm))
        # elif ((motion_p1 + motion_p2) < .08 and bpm > bpmLowerLimit):
        #     bpm -= 5.0
        #     AbletonTest.doSomething("/live/song/set/tempo " + str(bpm))
        # elif ((motion_p1 + motion_p2) < .12):
        #     #do nothing
        #     bpm += 0.0
        # elif ((motion_p1 + motion_p2) < .16 and bpm < bpmUpperLimit):
        #     bpm += 5.0
        #     AbletonTest.doSomething("/live/song/set/tempo " + str(bpm))
        # elif ((motion_p1 + motion_p2) < .20 and bpm < bpmUpperLimit):
        #     bpm += 10.0
        #     AbletonTest.doSomething("/live/song/set/tempo " + str(bpm))
            

        #3. proximity -> Zone
       
        
        if p1_detect == True:
            p1_zone = zonesDetect(centroid_p1)
        
        #print p1_zone and say p1_zone before printing
        print(f'p1_zone: {p1_zone}')
        
        if p2_detect == True:
            p2_zone = zonesDetect(centroid_p2)
            print(f'p2_zone: {p2_zone}')
            
            if (p1_zone == p2_zone) and (p1_zone != prevTrackZone):
                print(f'prevTrackZone: {prevTrackZone}')
                print(f'matchedZone: {p1_zone}')
                #set current track to new zone
                currentTrack = p1_zone
                #lower prev track volume to 0
                AbletonTest.doSomething("/live/track/set/volume " + str(prevTrackZone) + " .01")
                #change track to p1_zone
                AbletonTest.doSomething("/live/track/set/volume " + str(p1_zone) + " .75")
             
                prevTrackZone = p1_zone
                          
            
        ###################################
        ###################################
        ###################################

            

        #show the webcam frame that we just framed and annotated
        cv2.imshow("yolov8", frame)

        #break out it we esc
        if (cv2.waitKey(15) == 27):
            break
        
        #convert centroid_p1[0] to int
        #convert centroid_p1[1] to int

        x = int(centroid_p1[0])
        y = int(centroid_p1[1])
        

        # set circle position to the x and y coordinates
        circle_x = x
        circle_y = y

        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # draw circle and update screen
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, circle_color, (circle_x, circle_y), circle_radius)
        pygame.display.update()

if __name__ == "__main__":
    main()




