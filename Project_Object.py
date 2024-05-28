import requests
import json
from fuzzywuzzy import process
import pyaudio
import wave
import pyttsx3 
import datetime
import spacy
import re
from pyowm import OWM
import webbrowser
import pyttsx3.drivers.nsss
import numpy as np
import cv2
import time
import math

# This is the method where object detection takes place and bounding box is created for the intent(object of intrest) and specifies the position of the intent 
def object_detection(obj1):
           


    # Detecting Objects on Image with OpenCV deep learning library
    #
    # Algorithm:
    # Reading RGB image --> Getting Blob --> Loading YOLO v3 Network -->
    # --> Implementing Forward Pass --> Getting Bounding Boxes -->
    # --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels
    #
    # Result:
    # Window with Detected Objects, Bounding Boxes and Labels
    # Reading image with OpenCV library
    # In this way image is opened already as numpy array
    image_BGR = cv2.imread('images/4test.jpg')

    
    # Showing Original Image
    # Giving name to the window with Original Image
    # And specifying that window is resizable
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    # Pay attention! 'cv2.imshow' takes images in BGR format
    cv2.imshow('Original Image', image_BGR)
    # Waiting for any key being pressed
    cv2.waitKey(0)
    # Destroying opened window with name 'Original Image'
    cv2.destroyWindow('Original Image')

   

    # Getting spatial dimension of input image
    h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

   



    """
    Getting blob from input image
    """

    # Getting blob from input image
    # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob
    # from input image after mean subtraction, normalizing, and RB channels swapping
    
    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                                swapRB=True, crop=False)
    """
    Loading YOLO v3 network
    """

    # Loading COCO class labels from file
    # Opening file
   
    with open('datasets/coco.names') as f:
    #with open('yolo-coco-data/classes.names') as f:
        # Getting labels reading every line
        # and putting them into the list
        labels = [line.strip() for line in f]


   

    # Loading trained YOLO v3 Objects Detector
    # with the help of 'dnn' library from OpenCV
   
  

    network = cv2.dnn.readNetFromDarknet('datasets/yolov3.cfg',
                                        'datasets/final.weights')

    ##network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yo_test.cfg',
                                        #'yolo-coco-data/win.weights')

    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = network.getLayerNames()

 

    # Getting only output layers' names that we need from YOLO v3 algorithm
    # with function that returns indexes of layers with unconnected outputs
    layers_names_output = \
        [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

   

    # Setting minimum probability to eliminate weak predictions
    probability_minimum = 0.5

    # Setting threshold for filtering weak bounding boxes
    # with non-maximum suppression
    threshold = 0.3

    # Generating colours for representing every detected object
    # with function randint(low, high=None, size=None, dtype='l')
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    """
    Implementing Forward pass
    """

    # Implementing forward pass with our blob and only through output layers
    # Calculating at the same time, needed time for forward pass
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Showing spent time for forward pass
    print('Objects Detection took {:.5f} seconds'.format(end - start))

    


    """
    Getting bounding boxes
    """

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []
    inted=str(obj1)
    Gx = 0 # Gx is the variable to store the x co-ordinate of the centre of the intent object 
    Gy =0 # Gy is the variable to store the y co-ordinate of the centre of the intent object 
    md=0
    Ax= []
    Ay=[]
    Elab=[]
    Fstr=[]
    j=0
    index = 0
    obj2  = 0

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # # Check point
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial image size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original image and in this way get coordinates for center
                # of bounding box, its width and height for original image
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                
                
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
                
            



    """
    Non-maximum suppression
    """

    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                            probability_minimum, threshold)

    

    """
    Drawing bounding boxes and labels
    """

    # Defining counter for detected objects
    counter = 1

   
    if len(results) >0:
        # Going through indexes of results
        for i in results.flatten():
            counter += 1

            # Getting current bounding box coordinates,
            # its width and height
            
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = colours[class_numbers[i]].tolist()
            if (labels[int(class_numbers[i])]==inted):
                index = i
                
        
                cv2.rectangle(image_BGR, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(inted,
                                                    confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
            if (labels[int(class_numbers[i])]==obj2):
                if (labels[int(class_numbers[index])]==inted):
        
                    cv2.rectangle(image_BGR, (x_min, y_min),
                                (x_min + box_width, y_min + box_height),
                                colour_box_current, 2)

                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format(obj2,
                                                        confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
            
            #So in order to find the intent position it will compare the distance between  the intent object 
            #and all other detected objects(here centre co-ordinates of the object is considered for the 
            #distance comparisons) so the distance between nearby object detected will be less when 
            #compared to the other detected object hence the model will give the precise location of the intent object(object of intrest).

            
            if (labels[int(class_numbers[i])]==inted):
                
                
            
                        Gx=x_center
                        Gy=y_center
                        #print(Gx,Gy)
                
            else :
            
                Elab.append(labels[int(class_numbers[i])])
                Ax.append(x_min)
                Ay.append(y_min)
                #print(Ax[j],Ay[j])
                j = j+1

            for k in range(j) :

                p= [Gx,Gy]
                q= [Ax[k],Ay[k]]
                d= math.dist(p,q)
                                    
                if md==0:
                                        
                    md=d
                    Fx=Ax[k]
                    Fy=Ay[k]
                    obj2=Elab[k]
                   
                elif (d<md):
                                        
                    md=d
                    Fx=Ax[k]
                    Fy=Ay[k]
                    obj2=Elab[k]
                                    
            
                    #print('dist')
                    #print(md)
                #Note: Uncomment the below code in order to know the position of the object
                # if it is above, behind, right or left of the nearest object for more precision    
                # if Gx:
                #     Fstr.append(inted+' is left of '+str(obj))

                #     if Gy:
                #         Fstr.append(' and behind of '+str(obj))
                #     elif Gy>Fy:
                #         Fstr.append(' and above of '+str(obj))
                # elif Gx>Fx:
                #         Fstr.append(inted+' is right of '+str(obj))
                #         if Gy:
                #             Fstr.append(' and behind of '+str(obj))
                #         elif Gy>Fy:
                #             Fstr.append(' and above of '+str(obj))     

    # Comparing how many objects where before non-maximum suppression
    # and left after
    Fstr.append(inted +' is near to '+str(obj2))
    if labels[int(class_numbers[index])]==inted:
        print(*Fstr,sep='.')
        
    elif labels[int(class_numbers[index])]!=inted:
        Fstr.append("intent object is not detected or probably the model is not trained well ")
        Fstr[0] = Fstr[1]
      



    # Showing Original Image with Detected Objects
    # Giving name to the window with Original Image
    # And specifying that window is resizable
    cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
    cv2.imshow('Detections', image_BGR)
    # Waiting for any key being pressed
    cv2.waitKey(0)
    # Destroying opened window with name 'Detections'
    cv2.destroyWindow('Detections')
    return Fstr

#In this method we will record the user audio and generate the .wav file
def record_audio(RECORD_SECONDS, WAVE_OUTPUT_FILENAME):
    # --------- SETTING PARAMS FOR OUR AUDIO FILE ------------#
    FORMAT = pyaudio.paInt16  # format of wave
    CHANNELS = 1  # no. of audio channels
    RATE = 44100  # frame rate
    CHUNK = 1024  # frames per audio sample
    # --------------------------------------------------------#

    # creating PyAudio object
    audio = pyaudio.PyAudio()

    # open a new stream for microphone
    # It creates a PortAudio Stream Wrapper class object
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    # ----------------- start of recording -------------------#
    print("Listening...")

    # list to save all audio frames
    frames = []

    for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
        # read audio stream from microphone
        data = stream.read(CHUNK)
        # append audio data to frames list
        frames.append(data)

    # ------------------ end of recording --------------------#
    print("Finished recording.")

    stream.stop_stream()  # stop the stream object
    stream.close()  # close the stream object
    audio.terminate()  # terminate PortAudio

    # ------------------ saving audio ------------------------#

    # create wave file object
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

    # settings for wave file object
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))

    # closing the wave file object
    waveFile.close()
def read_audio(WAVE_FILENAME):
    # function to read audio(wav) file
    with open(WAVE_FILENAME, 'rb') as f:
        audio = f.read()
    return audio

# Wit speech API endpoint
API_ENDPOINT = 'https://api.wit.ai/speech'

# Wit.ai api access token
wit_access_token = 'XVG2CPKOSXYB2UULVV27OZFPUGEO6JMO'

# This method is used to recognize speech using wit.ai
def RecognizeSpeech(AUDIO_FILENAME, num_seconds):
    # record audio of specified length in specified audio file
    record_audio(num_seconds, AUDIO_FILENAME)

    # reading audio
    audio = read_audio(AUDIO_FILENAME)

    # defining headers for HTTP request
    headers = {'authorization': 'Bearer ' + wit_access_token,
               'Content-Type': 'audio/wav'}

    # making an HTTP post request
    resp = requests.post(API_ENDPOINT, headers=headers,
                         data=audio)

    # # converting response content to JSON format
    # data = json.loads(resp.content)
    try:
    # converting response content to JSON format
        data = json.loads(resp.content)
    except json.decoder.JSONDecodeError as e:
    # handle the error and return an empty string
        print("Error parsing JSON:", e)
        return ""
    # get text from data
    textt = data['text']

    # return the text
    return textt



# This method is to find the intent from the text obtained from speech
def fuzzy(sentence, list_of_items):
    str2match=sentence
    strOptions=list_of_items
    Ratios=process.extract(str2match,strOptions)
    highest=process.extractOne(str2match,strOptions)
    return highest


# This method is to find the time as requested by the user
def time_(list_of_text):
    now = datetime.datetime.now()
    return (now.hour,now.minute)


#This method gives the weather report as requested by user
def weather_(text):
    reg_ex = re.search('current weather in (.*)', text)

    cityy = reg_ex
    k = reg_ex
    x = reg_ex
    
    if reg_ex:
        cityy = reg_ex.group(1)
        api_key='ab0d5e80e8dafb2cb81fa9e82431c1fa'
        owm = OWM(api_key)
        obs = owm.weather_at_place(city)
        w = obs.get_weather()
        k = w.get_status()
        x = w.get_temperature(unit='celsius')
    return (cityy, k, x['temp_max'], x['temp_min'])


#This method is to open the website as requested by the user
def open_web(text):
        reg_ex = re.search('open (.+)',text )
        if reg_ex:
            domain = reg_ex.group(1)
            print(domain)
            url = 'https://www.' + domain+'.com'
            webbrowser.open(url)


#This method is to find the intent of the text       
def intent_finder(list_of_text, list_of_items):
    count=0
    l=list()
    obj=''
    # if l:
    # # sort the list and get the first element
    #  obj = sorted(l, key = lambda i: i[1], reverse = True)[0][0]
    # else:
    # # return an empty string if the list is empty
    #     obj = ""
    # print(list_of_text,'here printing')
    # print(list_of_items)
    for i in list_of_text:
        if i in list_of_items:
            obj=i
            test="the item you're looking for is "+""+i
            count=1
            break
    if (count==0):
        for i in list_of_text:
            k=fuzzy(i,list_of_items)
            l.append(k)
        obj = sorted(l, key = lambda i: i[1], reverse = True)[0][0]
        test="Do you mean"+" "+obj
        print("Do you mean : \n",obj)
    return test,obj
        





    
    

if __name__ == "__main__":  
    
        # text = RecognizeSpeech('v.wav', 10)
        text = 'where is book'
    
        print("\nYou said: {}".format(text))
        list_of_items=["keys","pen","book","watch","kettle","scissors", "bottle", "wine glass", "cup", "fork", "knife", "bowl","hair dryer","vase","tennis racket","laptop","person","eye glasses","chair","bed","mouse","mobile"]
        list_of_text=text.split()
       
        sp = spacy.load('en_core_web_sm')
        all_stopwords = sp.Defaults.stop_words
        list_of_text_without_sw= [word for word in list_of_text if not word in all_stopwords]
        # print(list_of_text_without_sw)
        test=""
        engine = pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate-50)
    
        if 'time' in list_of_text_without_sw:
            hour_,minutes_=time_(list_of_text_without_sw)
            # print('Current time is %d hours %d minutes' % (hour_,minutes_))
            test='Current time is %d hours %d minutes' % (hour_,minutes_)
            print('model:',test)
            engine.say(test)
            engine.runAndWait()
            engine = pyttsx3.init()
            
        elif 'hello' in list_of_text_without_sw or 'hi' in list_of_text_without_sw :
            # print("greetings")
            now = datetime.datetime.now()
            day_time = int(now.strftime('%H'))
            print(day_time)
            if day_time < 12:
                # print('Hello Sir.Good morning')
                test= 'Hi Good morning'
                print('model:',test)
                engine.say(test)
                engine.runAndWait()
                engine = pyttsx3.init()
            elif 12 <= day_time < 18:
               
                test= 'Hello  Good afternoon'
                print('model:',test)
                engine.say(test)
                engine.runAndWait()
                engine = pyttsx3.init()
            else:
                # print('Hello Sir. Good evening')
                test ='Hello Good evening'
                print('model:',test)
                engine.say(test)
                engine.runAndWait()
                engine = pyttsx3.init()
        elif 'weather' in list_of_text_without_sw:
        
            city, k, temp_max, temp_min = weather_(text)
            test = 'It is '+ str(temp_max)+'celsius and'+' '+ k +' '+ 'in'+' '+city 
            print('model:',test)
            engine.say(test)
            engine.runAndWait()
            engine = pyttsx3.init()
        elif 'open' in list_of_text_without_sw:
            open_web(text)
            test='The website you have requested has been opened for you .'
            print('model:',test)
            engine.say(test)
            engine.runAndWait()
            engine = pyttsx3.init()
            
    
        else:
            
         test,objj=intent_finder(list_of_text_without_sw,list_of_items)
         print('model:',test)   
         engine.say(test)
         engine.runAndWait()
         engine = pyttsx3.init()
         Final_output = object_detection(objj)
         pos__= Final_output[0]
         print('model:',pos__)
         engine.say(pos__)
         engine.runAndWait()
         engine = pyttsx3.init()
       