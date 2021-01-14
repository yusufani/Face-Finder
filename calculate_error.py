#%%




import pandas as pd
from sklearn.metrics import mean_squared_error
import cv2
import math
CSV_FILE_1 = "dataset/300W/01_Indoor/result.csv"
CSV_FILE_2 = "dataset/300W/02_Outdoor/result.csv"
#%%
df_1 = pd.read_csv(CSV_FILE_1)
df_2 = pd.read_csv(CSV_FILE_2)
frames = [df_1, df_2]
df = pd.concat(frames)
results = {}
#%%
def process_landmarks(landmarks):
    fields = landmarks.replace('(',')').split(")")
    datas= []
    for idx,val in enumerate(fields):
        if idx % 2 == 1 :
            fields = val.split(', ')
            datas.append((int(fields[0]) , int(fields[1]) ))
    return datas



for image_idx,data in df.iterrows():
    #print(type(data.landmarks))
    data.landmarks = str(data.landmarks)
    #print(data.landmarks)
    if data.landmarks != "null" and  data.landmarks != "nan" :
        landmarks = process_landmarks(data.landmarks)
        #print(landmarks)
        pts_file = data.path.replace(".png",".pts")
        with open(pts_file,"r") as fp:
            y_pred= []
            y_real  = []

            for idx,val in enumerate(fp.readlines()):
                if idx >=3 and idx < 71:
                    fields = val.split()
                    y_real.append(float(fields[0]))
                    y_real.append(float(fields[1]))

                    y_pred.append(float(landmarks[idx-3][0]))
                    y_pred.append(float(landmarks[idx-3][1]))
        error = mean_squared_error(y_real, y_pred, squared=False)

        image = cv2.imread(data.path)

        face_field = data.face_boxes.replace(")",'(').replace('(',',').split(',')
        face_field = int(face_field[3]) * int(face_field[4])
        error /= face_field
        time =  float(data.time)
    else:  # Not found faces
        error = math.sqrt(image.shape[0]**2 * image.shape[1]**2)
    print("Error: ", error)
    if results.get(data.detector,False)  == False:
        results[data.detector] = {}
    if results[data.detector].get(data.facial_landmarker, False) == False:
        results[data.detector][data.facial_landmarker] = {}
        results[data.detector][data.facial_landmarker] ["score"] = []
        results[data.detector][data.facial_landmarker] ["time"] = 0.0
        results[data.detector][data.facial_landmarker] ["count"] = 0
    results[data.detector][data.facial_landmarker] ["score"].append(error)
    if str(data.time) != 'nan' and str(data.time)!= 'null' and not data.time == "['null']" and not math.isnan(float(data.time)) :
        results[data.detector][data.facial_landmarker]["time"] += time
    else:
        results[data.detector][data.facial_landmarker]["time"] += results[data.detector][data.facial_landmarker]["time"] /results[data.detector][data.facial_landmarker]["count"]
    results[data.detector][data.facial_landmarker]["count"] +=1

#%%
# Normalize
min_x = 55555555
max_x = 0
for face in results.keys():
    for landmark in results[face]:
        #print('min : ' , min(results[face][landmark]['score']))
        if min(results[face][landmark]['score']) < min_x:
            min_x =  min(results[face][landmark]['score'])
        #print('max: ' , max(results[face][landmark]['score']))
        if max(results[face][landmark]['score']) > max_x:
            max_x = max(results[face][landmark]['score'])
#%%
for face in results.keys():
    for landmark in results[face]:
        total_error  = 0
        for error in results[face][landmark]['score']:
            total_error += (error- min_x) / (max_x - min_x)
        results[face][landmark]['score'] = total_error
#%%

X = []
plot_error = []
plot_time = []
for key, val in results.items():
    for key2, val2 in val.items():
        X.append(key+" +\n"+key2)
        plot_error.append(  val2['score'] / val2['count'])
        plot_time.append(  val2['time'] / val2['count'])
#%%
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 20, 5
#plt.ylim(0, 10000)
plt.xlabel('Models')

plt.ylabel('NMSE Errors')
plt.bar( X , plot_error, color= ('b','r', 'g' , 'b' , 'y' , 'purple','c'))
plt.show()
#%%
plt.xlabel('Models')
plt.ylabel('Passed time in seconds')
plt.bar(X,plot_time , color= ('b','r', 'g' , 'b' , 'y' , 'purple','c'))
plt.show()
#%%
plt.xlabel('Models')
plt.ylabel('NMSE Errors')
indexs = [0,1,3,4]
plt.bar( [ X[idx]  for idx in range(len( X))  if idx in indexs  ] ,[ plot_error[idx]  for idx in range(len( plot_error))  if idx in indexs  ] , color= ('b','r', 'b' , 'y' ))
plt.show()
#%%
plt.xlabel('Models')
plt.ylabel('Passed time in seconds')
plt.bar( [ X[idx]  for idx in range(len( X))  if idx in indexs  ] ,[ plot_time[idx]  for idx in range(len( plot_time))  if idx in indexs  ] , color= ('b','r', 'b' , 'y' ))

plt.show()
