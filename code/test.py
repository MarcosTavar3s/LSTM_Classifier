from lstm import classifier_model
import cv2
import numpy as np

model = classifier_model(dataset_dir="UCF50", classes_list=["WalkingWithDog","Skiing", "Swing", "Diving", "Mixing", "HorseRace", "HorseRiding"])

# Use the model to make predictions
model.load_model("model.keras")

# Output goes to 'output_video.mp4'
model.predict(video_file_path="UCF50/Swing/v_Swing_g04_c07.avi")

# Assemble a comparison between the prediction and the original video
cap_original = cv2.VideoCapture("UCF50/Swing/v_Swing_g04_c07.avi")
cap_predicted = cv2.VideoCapture("output_video.mp4")

width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
height =  int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_original.get(cv2.CAP_PROP_FPS))

output = cv2.VideoWriter('comparation.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps,
                         (2*width, height),
                         isColor=True) 

while cap_original.isOpened():
    ret_o, frame_original = cap_original.read()
    ret_p, frame_predicted = cap_predicted.read()
    frame_video = np.zeros((height, 2*width, 3), dtype=np.uint8)

    if not ret_o and not ret_p:
        break

    frame_video[:, :width, :] = frame_original[:,:,:]
    frame_video[:, width: , :] = frame_predicted[:,:,:]
    output.write(frame_video)

cap_original.release()
cap_predicted.release()
output.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture("comparasion.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret: 
        break
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    cv2.imshow("Comparation", frame)
    
cap.release()
cv2.destroyAllWindows()
