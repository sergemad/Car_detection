import streamlit as st
import numpy as np
import cv2 


#Setting Title of App
st.title("Car detection and count")
st.markdown("Upload an image with car")

#Uploading the dog image
image = st.file_uploader("Choose an image...")
submit = st.button('Predict car')
#On predict button click
if submit:


    if image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(opencv_image,(opencv_image.shape[1]*5,opencv_image.shape[0]*5))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        
        erosed = cv2.erode(blur,np.ones((2,2)), iterations = 6)
        
        dilated = cv2.dilate(erosed, np.ones((4,4)), iterations = 6)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
        
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        car_cascade_src = 'cars.xml'
        car_cascade = cv2.CascadeClassifier(car_cascade_src)
        cars = car_cascade.detectMultiScale(closing, 1.1, 1)

        cnt = 0
        for (x,y, w, h) in cars:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
            cnt += 1


        # Displaying the image
        #st.image(opening)
        st.image(img, channels="BGR")
        st.write("We found "+ str(cnt)+ " cars")

submit2 = st.button('Predict bus')
#On predict button click
if submit2:


    if image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(opencv_image,(opencv_image.shape[1]*5,opencv_image.shape[0]*5))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        
        erosed = cv2.erode(blur,np.ones((2,2)), iterations = 6)
        
        dilated = cv2.dilate(erosed, np.ones((4,4)), iterations = 6)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
        
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        bus_cascade_src = 'Bus_front.xml'
        bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
        bus = bus_cascade.detectMultiScale(closing, 1.1, 10)

        cnt2 = 0
        for (x,y, w, h) in bus:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
            cnt2 += 1


        # Displaying the image
        #st.image(opening)
        st.image(img, channels="BGR")
        st.write("We found "+ str(cnt2) + " bus")

