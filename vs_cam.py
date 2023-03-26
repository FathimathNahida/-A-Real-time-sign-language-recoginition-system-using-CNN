 #from predictionpage import predictcnn

from PIL import Image
from PIL import Image

#from predictionpage import predictcnn

output=["null","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
#
# vs = cv2.VideoCapture(0)
# while True:
#     ok, frame = vs.read()
#
#     if ok:
#         cv2image = cv2.flip(frame, 1)
#
#         x1 = int(0.5 * frame.shape[1])
#         y1 = 10
#         x2 = frame.shape[1] - 10
#         y2 = int(0.5 * frame.shape[1])
#
#         cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
#         cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
#
#         cv2.imshow('img', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
#             break

# Importing Libraries
# Importing Libraries
import cv2
import mediapipe as mp

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
	static_image_mode=False,

	min_detection_confidence=0.75,
	min_tracking_confidence=0.75,
	max_num_hands=2)

Draw = mp.solutions.drawing_utils
# listimg=[r"E:\sign language recognition - Copy\dataSet\trainingData\H\2.jpg",r"E:\sign language recognition - Copy\dataSet\trainingData\I\0.jpg",r"E:\sign language recognition - Copy\dataSet\trainingData\-1\1.jpg",
#         r"E:\sign language recognition - Copy\dataSet\trainingData\H\1.jpg",r"E:\sign language recognition - Copy\dataSet\trainingData\O\1.jpg",
#        r"E:\sign language recognition - Copy\dataSet\trainingData\W\1.jpg"]
listimg=[]
# def startcam():
countval = 0
# Start capturing video from webcam
cap = cv2.VideoCapture(0)
jj=0
txt=""
lastchar=""
while True:
    # Read video frame by frame
    _, frame = cap.read()
    try:
        img1=frame.copy()
        xx1 = int(0.5 * frame.shape[1])
        xy1 = 10
        xx2 = frame.shape[1] - 10
        xy2 = int(0.35 * frame.shape[1])

        cv2.rectangle(frame, (xx1 - 1, xy1 - 1), (xx2 + 1, xy2 + 1), (255, 0, 0), 1)
        cv2image = frame # cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # Flip image
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB image
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB image
        Process = hands.process(frameRGB)

        landmarkList = []
        # if hands are present in image(frame)
        if Process.multi_hand_landmarks:
            # detect handmarks
            for handlm in Process.multi_hand_landmarks:
                for _id, landmarks in enumerate(handlm.landmark):
                    # store height and width of image
                    height, width, color_channels = frame.shape

                    # calculate and append x, y coordinates
                    # of handmarks from image(frame) to lmList
                    x, y = int(landmarks.x * width), int(landmarks.y * height)
                    landmarkList.append([_id, x, y])

                # draw Landmarks
                Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)
        x1, y1, x2, y2 = 10000, 10000, 0, 0
        jj = jj + 1
        # If landmarks list is not empty
        if landmarkList != []:
            cv2image = cv2image[xy1: xy2, xx1: xx2]
            cv2.imwrite(r"H:\sign language recognition - Copy\sampleeee.jpg", cv2image)
            for i in range(0,len(landmarkList)):
                x,y=landmarkList[i][1], landmarkList[i][2]
                if(x1>x):
                    x1=x
                if x>x2:
                    x2=x

                if(y1>y):
                    y1=y
                if y>y2:
                    y2=y

            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2
            x1 = x1 - 10
            y1 = y1 - 10
            x2 = x2 + 10
            y2 = y2 + 10
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            # cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

            h = y2 - y1
            w = x2 - x1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # draw rectangle to main image

            # print(x1,y1,x2,y2,"=================================")
            #
            # print(x1,y1)
            try:
                print(jj)
                if jj >= 200:
                # print(y1,y2-y1,x1,x2-x1,"++++++++++++++++++++++++++++++++")


                    # cv2image = cv2.flip(frame, 1)
                    # cv2image = cv2image[y1 : y2, x1 : x2]
                    # cv2image = frame[ y1:(y1+h),x1:(x1+w)]
                    cv2.imwrite(r"H:\sign language recognition - Copy\sample.jpg",img1)
                    print("img1++++++++++===========================")
                    im = Image.open(r"H:\sign language recognition - Copy\sample.jpg")
                    im1 = im.crop( (x1,y1,x2, y2))
                    #
                    # # Shows the image in image viewer
                    im1 = im1.save(r"H:\sign language recognition - Copy\samplecrop1.jpg")
                    #
                    jj=0
                    image = cv2.imread(r"H:\sign language recognition - Copy\sampleeee.jpg")
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    invert = cv2.bitwise_not(gray)  # OR
                    # invert = 255 - image


                    # Setting parameter values
                    t_lower = 50  # Lower Threshold
                    t_upper = 150  # Upper threshold

                    # Applying the Canny Edge filter
                    edge = cv2.Canny(invert, t_lower, t_upper)

                    invert = cv2.bitwise_not(edge)  # OR

                    cv2.imwrite(r"H:\sign language recognition - Copy\samplecrop2.jpg",invert)
                    res=predictcnn(r"H:\sign language recognition - Copy\samplecrop2.jpg")
                    # res=predictcnn(r"H:\sign language recognition - Copy\dataSet\trainingData\-1\0.jpg")
                    print(res,"Result..................")

                    # if lastchar=="" and res=="0":
                    #     pass
                    # elif res=="0":
                    #     txt=txt+" "
                    #     lastchar=""
                    # else:
                    #     try:
                    #         lastchar=txt[-1]
                    #     except:
                    #         pass
                    #     txt=txt+res
                    if lastchar=="" and output[res[0]]=="null":
                        pass
                    elif output[res[0]]=="null":
                        txt=txt+" "
                        lastchar=""
                    else:
                        try:
                            lastchar=txt[-1]
                        except:
                            pass
                        txt=txt+output[res[0]]

            except Exception as e:
                print(e)
        else:
            image=frame
            # set brightness
            # sbc.set_brightness(int(b_level))

        # Display Video and when 'q' is entered,
        # destroy the window
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (350, 100)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2
        print("=======",txt)
        # Using cv2.putText() method
        cv2.putText(frame, txt, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Image', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            if txt != "":
               print("Word",txt)
            break
    except Exception as e:
        print(e)
        pass


# startcam()