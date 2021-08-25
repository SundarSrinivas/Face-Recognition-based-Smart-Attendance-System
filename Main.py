import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk
import csv
import pandas as pd
import datetime
import time
import os

# Graphical window
window = tk.Tk()  # to initialize the window
window.title("Face Recognition Based Attendance System") # to give the window a title
window.geometry('1280x720')   # shape of the window
dialog_title = ''
window.configure(background='grey') # configuring the window
window.rowconfigure(0, weight=1) # dividing the window into rows and columns according to their weight
window.columnconfigure(0, weight=1)
message = tk.Label(window, text='ATTENDANCE PORTAL', bg='silver', fg='black', width=20,
                   height=2, font=('helvetica', 20, 'bold'))
message.place(x=450, y=10)


# For the id od the student
lbl1 = tk.Label(window, text='ENTER ID', bg='silver', fg='black', width=15,
                height=1, font=('helvetica', 14, 'bold'))
lbl1.place(x=150, y=140)
txt1 = tk.Entry(window, width=20, bg='silver', fg='black',
                font=('arial', 14, 'bold'))
txt1.place(x=400, y=140)


# For the name of the student
lbl2 = tk.Label(window, text='ENTER NAME', bg='silver', fg='black',
                width=15, height=1, font=('helvetica', 14, 'bold'))
lbl2.place(x=150, y=200)
txt2 = tk.Entry(window, width=20, bg='silver', fg='black',
                font=('arial', 14, 'bold'))
txt2.place(x=400, y=200)


# For the notification
lbl3 = tk.Label(window, text='NOTIFICATION', bg='silver', fg='black',
                width=15, height=1, font=('helvetica', 14, 'bold'))
lbl3.place(x=150, y=260)
message3 = tk.Label(window, text='', bg='silver', fg='black', width=30,
                    height=4, font=('helvetica', 14, 'bold'))
message3.place(x=400, y=260)

# For the Attendance
lbl4 = tk.Label(window, text='ATTENDANCE', bg='silver', fg='black',
                width=15, height=1, font=('helvetica', 14, 'bold'))
lbl4.place(x=150, y=600)
message4 = tk.Label(window, text='', bg='silver', fg='black',
                    width=30, height=4, font=('helvetica', 14, 'bold'))
message4.place(x=400, y=600)




# Function for clearing the Id
def clearid():
    txt1.delete(0, 'end')
    res=""
    message3.configure(text= res)


# Function for clearing the name
def clearname():
    txt2.delete(0, 'end')
    res=""
    message3.configure(text = res)


# Function for taking the images
def TakeImages():
    Id = (txt1.get())
    name = (txt2.get())

    # Now check if Id is in number format and name is in String format
    if Id.isdigit() & (name.isalpha() or (' ' in name)):
        cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #ix=0
        sampleNum = 0

    # Starting the Webcam and taking the images
        while True:
            ret, fr = cam.read()
            print(ret)
            if ret:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(fr, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2)
                    #fc = fr[y:y + h, x:x + w, :]
                    sampleNum = sampleNum + 1
                    cv2.imwrite("TrainingImages\ " + name + "." + Id + "."
                                + str(sampleNum) + ".jpg",
                                gray[y:y + h, x:x + w])

                cv2.imshow('frame', fr)
                if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
                    break
                elif sampleNum > 100:  # Taking 61 sample images
                    break
            else:
                print('error')
                break

        cam.release()
        cv2.destroyAllWindows()
        res = "Images saved for Id:" + Id +'\n'+ " Name: " + name
        row = [Id, name]

        # Writing the contents in the csv file
        with open('Studentdetails\studentdetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message3.configure(text = res)
    # Message for the Notification box
    else:
        if Id.isdigit():
            res = "Enter the name in the correct format"
            message3.configure(text = res)

        if name.isalpha() or (' ' in name):
            res = " Enter the Id in correct format"
            message3.configure(text=res)


# Function for training the system
def trainImg():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face, Id = getImagesAndLabels("TrainingImages")
    recognizer.train(face, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainer.yml")
    res = "System Trained"  #+ ",".join(str(f) for f in Id)
    message3.configure(text = res)

def getImagesAndLabels(path):
    imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids =[]
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

# For tracking the images
def trackimg():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer.read("TrainingImageLabel\Trainer.yml")
    df = pd.read_csv("Studentdetails\studentdetails.csv")
    i=0
    cam= cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    font = cv2.FONT_HERSHEY_COMPLEX
    col_names = {'Id':[],'Name':[], 'Date':[],'Time':[] }
    attendance = pd.DataFrame(col_names)
    print(attendance)

    while True:
        ret, fr = cam.read()
        #print(ret)
        if ret:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)
                Id , conf = recognizer.predict(gray[y:y + h, x:x + w])
                if(conf<50):
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%D-%M')
                    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    ID = df.loc[df['Id'] == Id]['Id'].values
                    ID = str(ID)
                    ID = ID[1:-1]
                    bb = str(aa)
                    bb = bb[2:-2]
                    attendance = [str(ID), bb, str(date), str(timestamp)]
                    res = " Attendance Taken "
                    message3.configure(text=res)
                    res1 = "Student code: "+ ID + '\n' +  "Attendance taken successfully."
                    message4.configure(text=res1)

                else:
                    Id = 'Unknown'
                    bb = str(Id)

                cv2.putText(fr , str(bb), (x, y + h), font, 1, (255, 255, 255), 2)

            cv2.imshow('Taking Attendance', fr)
            if (cv2.waitKey(1) == ord('q')):
                break

        else:
            print('error')
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    exists = os.path.isfile("Attendance\Attendance_" + date + ".csv")
    if exists:
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()

    else:
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()

    csvFile1.close()
    cam.release()
    cv2.destroyAllWindows()


# Some used stuffs
global key
key = ''
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")

month ={'01':'January',
      '02':'February',
      '03':'March',
      '04':'April',
      '05':'May',
      '06':'June',
      '07':'July',
      '08':'August',
      '09':'September',
      '10':'October',
      '11':'November',
      '12':'December'
      }


# For clearing the id and the name
clear1 = tk.Button(window, text='CLEAR ID', command=clearid, bg='silver',
                   fg='black', width=15, height=1,
                   font=('helvetica', 14, 'bold'))
clear1.place(x=850, y=140)
clear2 = tk.Button(window, text='CLEAR NAME', command=clearname, bg='silver',
                   fg='black', width=15, height=1,
                   font=('helvetica', 14, 'bold'))
clear2.place(x=850, y=200)

# For Taking the images
takeimg = tk.Button(window, text='Take Images', command= TakeImages,
                    bg='silver', fg='black', width=15,
                    height=2, font=('helvetica', 14, 'bold'))
takeimg.place(x=150, y=400)

# For Training the images
trainimg = tk.Button(window, text='Train Images', command= trainImg,
                     bg='silver', fg='black', width=15,
                     height=2, font=('helvetica', 14, 'bold'))
trainimg.place(x=400, y=400)

# For Tracking the images
trackimg = tk.Button(window, text='Track Images', command= trackimg,
                     bg='silver', fg='black', width=15, height=2,
                     font=('helvetica', 14, 'bold'))
trackimg.place(x=650, y=400)

# For Closing the portal
quit = tk.Button(window, text='QUIT', command=window.destroy,
                 bg='silver', fg='black', width=15, height=2,
                 font=('helvetica', 14, 'bold'))
quit.place(x=900, y=400)

window.mainloop()
