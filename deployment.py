from ultralytics import YOLO
import streamlit as st
import numpy as np
import tempfile
import cv2
import os

class Deployment:
    def __init__(self):
        
        self.helmet_model = "models/ppe.pt"
        self.person_model = "models/person.pt"
        
        self.colors = [(0,255,0),(255,0,0),(0,0,255)]

    def detect_helmet_id(self, image, model):
        results = model.predict(image, conf=0.4, stream = True, device = "cuda:1")
        
        index = None
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                index = box.cls.astype(int)[0]
            break
        print(index)
        return index


    def run(self, video_path):
                
        person_model = YOLO(self.person_model)
        helmet_model = YOLO(self.helmet_model)
        
        FRAME_WINDOW = st.image([])
        
        video = cv2.VideoCapture(video_path)
        
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        class_ = {0: 'Hardhat', 1: 'No Hardhat'}
        
        while video.isOpened():
            try:
                ret, frame = video.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = person_model.predict(frame, conf=0.6, stream = True, classes = [0], device = "cuda:0")
                
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        points = box.xyxy[0].astype(int)
                        xmin,ymin,xmax,ymax = points
                                                
                        person = frame[max(0, ymin-30):ymax,max(0, xmin-20):min(width, xmax+20)]
                        person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                        
                        index = self.detect_helmet_id(person, helmet_model)
                        
                        if index is not None:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), self.colors[index], 2)
                            cv2.putText(frame, class_[index], (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[-1], 2, cv2.LINE_AA)
                        
                if not ret or video.isOpened()==False:
                    break

                FRAME_WINDOW.image(frame)
                
            except:
                break
        video.release()


if  __name__ == '__main__':
    
    st.title("Hardhat(Helmet) detection")
    src = st.radio("Select video type: ", ["Sample video", "Upload video"])
    deployment = Deployment()
    
    if src=="Sample video":
        deployment.run("sample.mp4")

    else:
        file = st.file_uploader("Upload your file here...", type=["mp4", "avi", "mov"]) 
        if file is not None:
            video_path = None
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                video_path = temp_file.name
                temp_file.write(file.read())
                deployment.run(video_path)
            os.remove(video_path)
