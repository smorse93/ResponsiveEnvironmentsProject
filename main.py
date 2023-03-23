#annotator stuff
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

#general
from operator import index
import cv2
import argparse
#from torch import true_divide
from ultralytics import YOLO
import supervision as sv
import numpy as np
import math

import sys
sys.path.insert(0, './Ableton/User Library/Remote Scripts/AbletonOSC')
import AbletonTest


#getting ByteTrack and other libs working
#ignore for now, not sure if I need ByteTrack
import sys
sys.path.append(f"/ByteTrack")

from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
# from onemetric.cv.utils.iou import box_iou_batch


colors = sv.ColorPalette.default()

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




# ------------- UTILITES -------------------------

# geometry utilities
@dataclass(frozen=True)
class Point:
    x: float
    y: float
    
    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x
    
    @property
    def min_y(self) -> float:
        return self.y
    
    @property
    def max_x(self) -> float:
        return self.x + self.width
    
    @property
    def max_y(self) -> float:
        return self.y + self.height
        
    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)
    
    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float) -> Rect:
        return Rect(
            x=self.x - padding, 
            y=self.y - padding,
            width=self.width + 2*padding,
            height=self.height + 2*padding
        )
    
    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y

@dataclass
class Detection:
    rect: Rect
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None

    @classmethod
    def from_results(cls, pred: np.ndarray, names: Dict[int, str]) -> List[Detection]:
        result = []
        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            class_id=int(class_id)
            result.append(Detection(
                rect=Rect(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                ),
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidence)
            ))
        return result


def filter_detections_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
    return [
        detection
        for detection 
        in detections
        if detection.class_name == class_name
    ]
# draw utilities


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int
        
    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

    @classmethod
    def from_hex_string(cls, hex_string: str) -> Color:
        r, g, b = tuple(int(hex_string[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return Color(r=r, g=g, b=b)


def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image


def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, -1)
    return image


def draw_polygon(image: np.ndarray, countour: np.ndarray, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, thickness)
    return image


def draw_filled_polygon(image: np.ndarray, countour: np.ndarray, color: Color) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, -1)
    return image


def draw_text(image: np.ndarray, anchor: Point, text: str, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.putText(image, text, anchor.int_xy_tuple, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.bgr_tuple, thickness, 2, False)
    return image


def draw_ellipse(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4
    )
    return image


# base annotator
  

@dataclass
class BaseAnnotator:
    colors: List[Color]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_ellipse(
                image=image,
                rect=detection.rect,
                color=self.colors[detection.class_id],
                thickness=self.thickness
            )
        return annotated_image

#area of bounding box
def area (p1_loc, p2_loc):
    area_p1 = np.array(abs(p1_loc[0][2] - p1_loc[0][0])*abs(p1_loc[0][3] - p1_loc[0][1]))
    area_p2 = np.array(abs(p2_loc[0][2] - p2_loc[0][0])*abs(p2_loc[0][3] - p2_loc[0][1]))
    return area_p1, area_p2


#centroid finding ---THis is incorrect 
def centroid (p1_loc, p2_loc):
    diff_xy_p1 = np.array([p1_loc[0][2] - p1_loc[0][0],p1_loc[0][3] - p1_loc[0][1]])
    diff_xy_p2 = np.array([p2_loc[0][2] - p2_loc[0][0],p2_loc[0][3] - p2_loc[0][1]])
    centroid_p1 = np.array([(p1_loc[0][0]+diff_xy_p1[0]), (p1_loc[0][1]+diff_xy_p1[1])])
    centroid_p2 = np.array([(p2_loc[0][0]+diff_xy_p2[0]), (p2_loc[0][1]+diff_xy_p2[1])])
    return centroid_p1, centroid_p2

#distance
def distance (centroid_p1, centroid_p2):
    dist = math.sqrt( ((centroid_p1[0][0]-centroid_p2[0][0])**2) + ((centroid_p1[0][1]-centroid_p2[0][1])**2))
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

#--------------------- MAIN ----------------------
def main():
    #getting webcam to run
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

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
    p1_loc = np.array([[0, 0, 0, 0]])
    p2_loc = np.array([[0, 0, 0, 0]])
    centroid_p1 = None
    centroid_p2 = None
    centroid_p1_prev = None
    centroid_p2_prev = None
      

    ########################################################
    ########################################################
    
    #DO SOMETHING WITH ABLETONTEST
    #ex --> command_str = "/live/song/set/tempo 123.0"
    #ex --> command_str = "/live/song/set/tempo 124.0"
    
#    AbletonTest.doSomething("/live/song/set/tempo 150.0")
    
    ########################################################
    ########################################################
    
    
    while True:
        #read the current frame
        ret, frame = cap.read()

        #put frame into model
        result = model(frame, agnostic_nms=True)[0]
        #get detections using the model
        detections = sv.Detections.from_yolov8(result)
       
        #limit detections to mask area, class id people, and confidence >0.5 
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
        a = 0
        p1_detect = False
        p2_detect = False
        
        #for loop
        for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = zone_annotator.annotate(scene=frame)
#            print(f'index: {a}')
#            print(detections_filtered.xyxy)
        
            if detections_filtered.xyxy.any():
                #assign p1 and p2 
                if p1_detect == False:
                    p1_detect = True
                else: 
                    p2_detect = True

                #assign p1 and p2 locations
                if p1_detect == True and p2_detect == False:
                    p1_loc = detections_filtered.xyxy
                else: 
                    p2_loc = detections_filtered.xyxy
                    
#                print(f'object detected in zone: {a}')
                
                
            a = a+1 

        
        
        print(f'p1_detect{p1_detect}')   
        print(f'p2_detect{p2_detect}')
        
        print(f'p1_loc{p1_loc}')
        print(f'p2_loc{p2_loc}')

        if (centroid_p1 is not None):
            centroid_p1_prev = centroid_p1
            centroid_p2_prev = centroid_p2

        centroid_p1, centroid_p2 = centroid(p1_loc,p2_loc)
        print("centroid_p1:")
        print(centroid_p1)
        print("centroid_p2:")
        print(centroid_p2)

        dist = distance(p1_loc,p2_loc)
        print('distance:')
        print(dist)
        
        if (dist < 200):
            AbletonTest.doSomething("/live/song/set/tempo 80.0")
        elif dist < 400:
            AbletonTest.doSomething("/live/song/set/tempo 100.0")
        elif dist < 600:
            AbletonTest.doSomething("/live/song/set/tempo 120.0")
        else:
            AbletonTest.doSomething("/live/song/set/tempo 140.0")
            

        if (centroid_p1_prev is not None):
            motion_p1, motion_p2 = motion(centroid_p1, centroid_p1_prev, centroid_p2, centroid_p2_prev)
#            print("motion_p1:")
#            print(motion_p1)
#            print("motion_p2:")
#            print(motion_p2)


        #create a dict for all of our labels
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]  
#        print("labels:")
#        print(labels)

        #show the webcam frame that we just framed and annotated
        cv2.imshow("yolov8", frame)

        #break out it we esc
        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()




