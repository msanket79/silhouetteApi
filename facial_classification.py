#install requirements by:
#           pip install -r requirements.txt
# this module prepares the dataset to feed into the model
# it will be using python h5py file to create a dataset and then the file would be loaded into pytorch using pytorch dataloader

# Requirements: 
# there should be a directory train like this
# data/
#     train/
#         person1/
#             photo1.jpg (photos could be of any name)
#                 .
#                 .
#             photoN.jpg
#         person2/
#             photo1.jpg (photos could be of any name)
#                 .
#                 .
#             photoN.jpg
#           .
#           .
#           .
#           .
#         personN/
#             photo1.jpg (photos could be of any name)
#                 . 
#                 .
#             photon.jpg

# another directory where model - related files would be stored is (you can specify where to save this folder)
# no files of this directory have to be created by user.

# model_files/
#     data/
#         batch1.h5py              this is the trainset - contains embeddings and their labels
#         batch2.h5py              batch2['X'][idx] -> embeddings of idx      batch2['y'][idx] -> label of idx embeddings 
#             .
#             .
#         batchN.h5py

#         user.h5py              user-idx mapping. by default the model will return an idx, each user has a unique index


#     model/
#         trained_classifier     this is the trained classifier and it will be restored from here, in case gone the user has to re train the model


# classifier.train(dir_path, learning_rate,num_epoch,reset_model)

#     calling the train function with dataset folder parameter would train it. train function also has an extra argument reset_model where it is going to train a new fresh model from scratch
#     also the model will not train unless there is at least one photo in the directory in the specified format
#     train function can also be supplied with parameters of num_epochs, learning_rate

# classifier.remove_data(batch)

#     this function would remove the batch's data and delete all the users of the batch from the user_idx mapping.
#     since this is directly going to affect the model's output classes, user has to retrain the model after this step.
#     THE MODEL IS GOING TO FORGET EVERYTHING AFTER THIS STEP!! RETRAIN IT


# classifier.test(img_path, num_samples = 1)

#     the first argument is an image, if supplied- then model will try to classify that image - [formats supported : all formats supported by pillow and base64 encodings]
#     if no image is supplied, then camera is going to open and once it detects a face in the video with probablity >= face_detection_threshold [0.9 by default] and then it is going to classify that face
#     the second process is going to be repeated num_samples times, i.e. num_samples times a face is going to be searched and classified and results would be average over all results
#     results:
#         will return a dictionary of size 2 
#           dict{user : prob}       where user is the label with probablity prob.
#         the 2 items are going to be the top 2 labels with the highest probablity
#     also, the test function is also going to store the labeled face as lastentry/lastentry.jpg


#if classifier.train() returns 100, it means that the data/train directory is in incorrect format
#if classifier.test() returns 106, it means that the model is not trained and testing is attempted

import torch
import cv2
import dlib
import h5py
from pathlib import Path
import base64
import pyheif
from facenet_pytorch import MTCNN,InceptionResnetV1
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
import numpy as np

class classifier:
    def __init__(self,model_files_path = 'model_files',data_files_path ="data",liveness_check = True, liveness_ear_threshold = 0.21,face_detection_threshold = 0.9,USE_GPU = False):
        self.device = torch.device('cuda') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')
        self.data_files_path = Path(data_files_path)
        self.model_files_path = Path(model_files_path)
        self.face_detection_threshold = face_detection_threshold
        
        #initialising face detector
        # mtcnn_path = self.model_files_path / "saved_models/mtcnn.onnx"
        # if mtcnn_path.exists():
        #     self.mtcnn = cv2.dnn.readNetFromONNX(str(mtcnn_path))            
        # else:
        #     self.mtcnn = None

        self.mtcnn = MTCNN(device = self.device)
        self.mtcnn_ = MTCNN(post_process = False,device = self.device)
        #initialising embedding_generator
        # inception_path = self.model_files_path / "saved_models/InceptionResnetV1.onnx"
        # if inception_path.exists():
        #     self.embedding_generator = cv2.dnn.readNetFromONNX(str(inception_path))            
        # else:
        #     self.embedding_generator = None
        self.embedding_generator = InceptionResnetV1(pretrained = 'vggface2').eval()

        #checking if all directories exist
        check_paths = [(self.data_files_path / "train"),(self.model_files_path/"saved_models"),(self.model_files_path/"data")]
        for path in check_paths:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        
        #storing all user mappings
        self.user_idx_mapping = {}
        self.idx_user_mapping = {}
        self.getMappings()

        #generating batch_list
        self.batch_list = set()
        if self.idx_user_mapping.__len__()>0:
            for key,value in self.idx_user_mapping.items():
                self.batch_list.add(value[:2])  #21bcs106 -> 21

        #liveness checker - basic blink tester
        if liveness_check:
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        self.liveness_ear_threshold = liveness_ear_threshold
        
        #loading the svm
        self.SVM_classifier = None
        svm_path = self.model_files_path / "saved_models/SVM.pkl"
        if svm_path.exists():
            with open(str(svm_path), 'rb') as f:
                self.SVM_classifier = pickle.load(f)

        else:
            self.trainSVM()    

    def removeBatch(self,batch:str):
        self.idx_user_mapping.clear()
        self.user_idx_mapping.clear() #dictionary with key as user and idx as value

        #deleting the batch's data:
        (self.model_files_path/"data"/ (batch+".h5py")).unlink(missing_ok=True)

        #removing the batch roll no. from the mapping
        self.getMappings(block_batch=batch,get_user_idx_map=True)
         
        #updating user File
        (self.model_files_path/"data/users.h5py").unlink(missing_ok=True)


        #creating new users file based on mapping
        num_users = len(self.user_idx_mapping)
        with h5py.File(self.model_files_path/'data/users.h5py','w') as f:
            f.create_dataset('label',(num_users,1),'S10')
            for key in self.user_idx_mapping:
                label = key
                idx = self.user_idx_mapping[key]
                f['label'][idx] = label
        
        #retrain the svm
        self.trainSVM()

    def addData(self, test_img, label):

        test_imgs = []
        labels = []
        
        if isinstance(test_img,str):
        #if it is single image
            test_imgs.append(test_img)
            labels.append(label)
        else:
        #if there are mutliple
            test_imgs = test_img
            labels = label
        try:
            for idx in range(len(test_imgs)):
                im_path = Path(test_img)

                user_folder = self.data_files_path/"train"/labels[idx]
                #creating directory for the user
                user_folder.mkdir(parents=True, exist_ok=True)

                new_impath = user_folder/im_path.name

                #moving to train dir
                im_path.rename(new_impath)
            return True
        except:
            return False

    def train(self,batch_size = 1):
        dir_path = self.data_files_path/"train"

        self.updateUserFile()

        #checking if the directories are in the proper format and/or have images   
        self.num_images = len(list(dir_path.glob("*/*.jpg")))
        if self.num_images == 0:
            return 100 

        # creating user to idx mapping
        for key,value in self.idx_user_mapping.items():
            self.user_idx_mapping[value] = key

        #updating the user mapping in modelfiles/user.h5py
        self.updateDataFile(batch_size)

        self.trainSVM()

    #function to update users.h5py
    def updateUserFile(self):

        dir_path = self.data_files_path /"train"
        users = []

        for dir in dir_path.iterdir():
            if dir.is_dir():
                users.append(str(dir.name))

        #user index mapping will be stored here first to avoid inconsistency and duplicates
        self.user_idx_mapping.clear() #dictionary with key as user and idx as value
        self.idx_user_mapping.clear()
        filelength = 0

        #checking for new data and creating their mappings
        self.getMappings(get_user_idx_map=True)
        filelength = len(self.user_idx_mapping)
        # adding new users
        for user in users:
            if not user in self.user_idx_mapping:
                self.user_idx_mapping[user] = filelength
                self.idx_user_mapping[filelength] = user
                filelength += 1
                
        #updating user File
        (self.model_files_path/"data/users.h5py").unlink(missing_ok=True)

        num_users = len(self.user_idx_mapping)
        with h5py.File(self.model_files_path/'data/users.h5py','w') as f:
            f.create_dataset('label',(num_users,1),'S10')
            for key in self.user_idx_mapping:
                label = key
                idx = self.user_idx_mapping[key]
                f['label'][idx] = label

        #user file is successfully updated
        #updating the batch list
        self.batch_list.clear()
        if self.idx_user_mapping.__len__()>0:
            for key,value in self.idx_user_mapping.items():
                self.batch_list.add(value[:2])  #21bcs106 -> 21
    
    def updateDataFile(self,batch_size=1):
        dir_path = self.data_files_path/"train"
        #this will loop for all users batch-wise
        all_users_list = []

        if self.user_idx_mapping.__len__() == 0:
            for key,values in self.idx_user_mapping.items():
                self.user_idx_mapping[values] = key

        for dir in dir_path.iterdir():
            if dir.is_dir():
                all_users_list.append(str(dir.name))

        imgs_batch = []
        total_added_embeddings = 0
        for batch in self.batch_list:
            num_images = len(list(dir_path.glob(batch+'*/*')))
            cur_data_idx = 0
            dataFile = None
            if (self.model_files_path/'data'/(batch+".h5py")).exists():
                dataFile = h5py.File(self.model_files_path/'data'/(batch+'.h5py'),'a')
                cur_data_idx = len(dataFile['y'])
                old_size = cur_data_idx
                new_size = old_size + num_images
                dataFile['X'].resize((new_size,512))
                dataFile['y'].resize((new_size,1))
            else:
                dataFile = h5py.File(self.model_files_path/'data'/(batch+'.h5py'),'w')
                dataFile.create_dataset('X',(num_images,512),maxshape=(None,512)) 
                dataFile.create_dataset('y',(num_images,1), maxshape=(None,1))
                    
            for user in all_users_list:
                if(user[0:2] != batch):
                    continue
                user_idx = self.user_idx_mapping[user]
                img_paths = list((dir_path/user).glob("*"))
                for img_path in img_paths:
                    img = self.openImg(img_path)
                    imgs_batch.append(img)
                    del(img)
                    if len(imgs_batch) == batch_size:
                        imgs_batch_tensor = torch.from_numpy(np.stack(imgs_batch))
                        embeddings = self.getEmbedding(imgs_batch_tensor)
                        #updating in file:
                        for embedding in embeddings:
                            dataFile['X'][cur_data_idx] = embedding
                            dataFile['y'][cur_data_idx] = user_idx
                            cur_data_idx += 1
                        del(embeddings)
                        del(imgs_batch_tensor)
                        del(imgs_batch[:])
                    
                #if the user loop is finished i.e. the batch size is more than the total items:
                if len(imgs_batch) > 0:
                    imgs_batch_tensor = torch.from_numpy(np.stack(imgs_batch))
                    embeddings = self.getEmbedding(imgs_batch_tensor)
                    #updating in file:
                    for embedding in embeddings:
                        dataFile['X'][cur_data_idx] = embedding
                        dataFile['y'][cur_data_idx] = user_idx
                        cur_data_idx += 1
                    del(embeddings)
                    del(imgs_batch_tensor)
                    del(imgs_batch[:])
            total_added_embeddings += len(dataFile['y'])
            del(dataFile)
        print(f"CREATED {total_added_embeddings} EMBEDDINGS")      
    
    def testSVM(self,img_path = None,check_liveness = False,device = 0,num_samples=1):
        if self.SVM_classifier == None:
            return 106

        camera_mode = False if img_path else True
    

        #checking if the argument is 1 photo or multiple photos
        imgs_batch = []

        if camera_mode == False:
        # creating a list of all photos
            test_imgs = []

            if isinstance(img_path,str):
                test_imgs.append(img_path)
            else:
                test_imgs = img_path

            for img_path in test_imgs:
                imgs_batch.append(self.openImg(img_path))
            
                
        else:
            try:
                for sample in range(num_samples):  
                    vid = cv2.VideoCapture(device)
                    prob = 0.0
                    first_open = False
                    second_closed = False
                    third_open = False
                    blink = False
                    while(True):
                            
                        # Capture the video frame
                        # by frame
                        ret, test_img = vid.read()
                    
                        # Display the resulting frame
                        # cv2.imshow('frame', frame)
                        with torch.no_grad():
                            face,prob = self.mtcnn(test_img,return_prob = True)

                        #if a face is found with >90% probab
                        if prob and prob>=self.face_detection_threshold:
                            if check_liveness == True:
                                # try:
                                if blink == True:
                                    imgs_batch.append(test_img)
                                    break
                                else:
                                    status,score = self.livenessCheck(test_img)
                                    if first_open == False:
                                        if status == False:
                                            first_open = True
                                            print(f"first open : {first_open}")
                                    elif second_closed == False:
                                        if status == True:
                                            second_closed = True
                                            print(f"second closed : {second_closed}")
                                    elif third_open == False:
                                        if status == False:
                                            third_open = True
                                            print(f"third open : {third_open}")
                                    blink = first_open and second_closed and third_open
                                # except:
                                #     pass
                            else:
                                imgs_batch.append(test_img)
                                break
                    # After the loop release the cap object
                    vid.release()
                    # Destroy all the windows
                    cv2.destroyAllWindows()
                
            finally:
                # After the loop release the cap object
                vid.release()
                # Destroy all the windows
                cv2.destroyAllWindows()
        imgs_batch_tensor = torch.from_numpy(np.stack(imgs_batch))
        embeddings = self.getEmbedding(imgs_batch_tensor)
        if embeddings is None:
            del(embeddings)
            return []
        y_pred = self.SVM_classifier.predict(embeddings)
        del(embeddings)
        del(imgs_batch_tensor)
        face = self.mtcnn_(imgs_batch[-1])
        del(imgs_batch[:])


        Path("lastentry").mkdir(parents=True,exist_ok=True)
        
        # cv2.imwrite('lastentry/lastentry.jpg',face)   

        labels, counts  = np.unique(y_pred,return_counts = True)
        idx = np.argsort(counts)
        actual_labels = []
        try:
            actual_labels.append(self.idx_user_mapping[labels[idx[0]]])
        except:
            pass
        if len(idx)>1:
            actual_labels.append(self.idx_user_mapping[labels[idx[1]]])
        return actual_labels
    
    def getMappings(self,block_batch=None, get_user_idx_map = False):
        users_file = self.model_files_path / "data/users.h5py"
        
        if not users_file.exists():
            return
        try:
            with h5py.File(users_file,'r') as f:
                cur_idx = 0
                for i in range(len(f['label'])):
                    key = str(f['label'][i])  #[b'21bcs106']
                    key = key[3:-2]
                    if block_batch and key[:2] == block_batch:
                        continue
                    self.idx_user_mapping[cur_idx] = key
                    if get_user_idx_map:
                        self.user_idx_mapping[key] = cur_idx
                    cur_idx += 1
            self.out_features = len(self.idx_user_mapping)
        except:
            pass
    
    def openImg(self,test_img:str=None):
        if test_img is None:
            return
        img_path = Path(test_img)
        if img_path.suffix == '.heic':
            format = "heic"
        elif img_path.suffix == '.jpg' or img_path.suffix == '.jpeg' or img_path.suffix == '.png':
            format = "jpgpng"
        else:
            format = "base64"

        try:
            test_img = str(test_img)
            if format == "base64":
                test_img = test_img.split(',')[1]
                decoded_image_data = base64.b64decode(test_img)

                # Convert the image data to a NumPy array
                image_array = np.frombuffer(decoded_image_data, dtype=np.uint8)

                    # Read the image with OpenCV
                test_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
                
            elif format == "jpgpng":
                test_img = cv2.imread(test_img)
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
                
            elif format == "heic":
                heif_file = pyheif.read(test_img)
                width = heif_file.size[0]
                height = heif_file.size[1]
                test_img = np.array(heif_file.data)
                test_img = test_img.reshape((height, width, 3))
            # height, width, channels = test_img.shape
            # aspect_ratio = width / height
            # new_width = 480
            # new_height = int(new_width / aspect_ratio)
            test_img = cv2.resize(test_img,(640,480))
            return test_img
        except:
            print("could not open image")

    def getEmbedding(self,test_img, numpy = True):    #accepting
        try:
            with torch.no_grad():
                faces = self.mtcnn(test_img)
                faces = torch.stack(faces).to(self.device)
                embeddings = self.embedding_generator(faces)
                if numpy:
                    embeddings = embeddings.to(torch.device("cpu")).detach().numpy()
        except:
            embeddings = None #no face found
        return embeddings

    def trainSVM(self):
        #right now trying the hard coded values:
        new_classifier = LinearSVC()
        #code for training the best svm
        
        # Load the dataset

        # iris = datasets()
        # X = iris.data
        # y = iris.target

        # Split the dataset into training and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Define the parameter distribution for SVC
        # param_dist = {'C': np.logspace(-4, 4, 20),
        #             'gamma': np.logspace(-4, 4, 20),
        #             'kernel': ['linear', 'rbf', 'poly']}

        # Create an SVC object
        # svc = SVC()

        # Create a RandomizedSearchCV object and fit it to the training data
        # random_search = RandomizedSearchCV(svc, param_distributions=param_dist, n_iter=100, random_state=42, cv=2)
        # random_search.fit(X_train, y_train)

        # Print the best parameters and score
        # print("Best parameters: ", random_search.best_params_)
        # print("Best score: ", random_search.best_score_)

        # svc_best = SVC(C=random_search.best_params_['C'],
        #             gamma=random_search.best_params_['gamma'],
        #             kernel=random_search.best_params_['kernel'])

        
        y_train = []
        x_train =[]
        for batch in self.batch_list:
            dir = "model_files/data/"+batch+".h5py"
            with h5py.File(dir,'r') as f:
                for i in range(len(f['y'])):
                    x_train.append(f['X'][i])
                    y_train.append(f['y'][i])
        new_classifier.fit(x_train,y_train)
        self.SVM_classifier = new_classifier
        del(new_classifier)
        with open('model_files/saved_models/SVM.pkl', 'wb') as f:
            pickle.dump(self.SVM_classifier, f)

    def eye_aspect_ratio(self,eye):
        # Compute the Euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])

        # Compute the Euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye[0] - eye[3])

        # Compute the aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear
    
    def livenessCheck(self,frame): #returns if eyes are closed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray)
        for face in faces:
            landmarks = self.dlib_predictor(gray, face)

            # Extract the coordinates of the left and right eye landmarks
            left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)])
            right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                (landmarks.part(43).x, landmarks.part(43).y),
                                (landmarks.part(44).x, landmarks.part(44).y),
                                (landmarks.part(45).x, landmarks.part(45).y),
                                (landmarks.part(46).x, landmarks.part(46).y),
                                (landmarks.part(47).x, landmarks.part(47).y)])

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)

            # Compute the average aspect ratio for both eyes
            ear = (left_ear + right_ear) / 2.0
            return ear<self.liveness_ear_threshold, ear
