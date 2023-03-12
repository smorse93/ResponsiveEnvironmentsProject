from operator import index
import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

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
        for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = zone_annotator.annotate(scene=frame)
            print(f'index: {a}')
            print(detections)
            
            a = a+1 


        #create a dict for all of our labels
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]  
        
        #show the webcam frame that we just framed and annotated
        cv2.imshow("yolov8", frame)

        #break out it we esc
        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()

