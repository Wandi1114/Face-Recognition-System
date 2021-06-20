import cv2
import os
import numpy as np

folder_name = 'faces data'                                         # Prepare new folder do PC kalian
os.mkdir(folder_name)
os.mkdir('recognizer')

face_cascade_file = 'Cascade Classifier/face-detect.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_file)            # Load cascade classifiernya

total_images = 5
counter = 1
ids = 1

cam = cv2.VideoCapture(0)                                           # Akses Kamera
while True:
    ret, frame = cam.read()                                         # Membaca setiap frame dari stream kamera
    frame_copy = frame.copy()                                       # Copy frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                  # Mengubah mode BGR ke GRAY (hitam putih)
    
                                                                    # Proses pencarian wajah 
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)             # <cascade_file>.detectMultiScale(<frame>, <scale_factor>, <min_neighbors>)
    for x, y, w, h in faces:                                        # Looping semua wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)    # Gambar box untuk setiap wajah
        
        if cv2.waitKey(1) & 0xff == ord('c'):                       # Menunggu tombol c di tekan
            roi_face = frame_copy[y:y+h, x:x+w]                     # region of interest dari frame
            cv2.imwrite(f'{folder_name}/people.{ids}.{counter}.jpg',# write region wajah
                        roi_face)
            
            counter += 1
            if counter > total_images:
                print(f'[INFO] {total_images} IMAGE CAPTURED!')     # info done proses
        
    cv2.imshow('Face Detect Video', frame)                          # Jendela untuk menampilkan hasil
    
    if cv2.waitKey(1) & 0xff == ord('x'):                           # Exit dengan tombol x
        break
        
cam.release()                                                       # Menyudahi akses kamera
cv2.destroyAllWindows()                                             # Menutup jendela