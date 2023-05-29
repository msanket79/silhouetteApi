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
#         trained_classifier     this is the trained classifier and it will be restored from here, in case gone, the user has to re train the model


# classifier.train()

#     calling the train function with dataset folder parameter would train it. 
#     also the model will not train unless there is at least one photo in the directory in the specified format

# classifier.remove_data(batch)

#     this function would remove the batch's data and delete all the users of the batch from the user_idx mapping.
#     since this is directly going to affect the model's output classes, user has to retrain the model after this step.
#     THE MODEL IS GOING TO FORGET EVERYTHING AFTER THIS STEP!! RETRAIN IT


# classifier.test(img_path, num_samples = 1)

#     the first argument is an image, if supplied- then model will try to classify that image - [formats supported : all formats supported by openCV,heic and base64 encodings]
#     if no image is supplied, then camera is going to open and once it detects a face in the video with probablity >= face_detection_threshold [0.9 by default] and then it is going to classify that face
#     the second process is going to be repeated num_samples times, i.e. num_samples times a face is going to be searched and classified and results would be average over all results
#     results:
#         will return a list of size 0,1 or 2 depending on the top people classified as that photo by the model
#     also, the test function is also going to store the labeled face as lastentry/lastentry.jpg


#if classifier.train() returns 100, it means that the data/train directory is in incorrect format
#if classifier.test() returns 106, it means that the model is not trained and testing is attempted

import torch
import cv2
import dlib
import h5py
from collections import defaultdict
import psycopg2
from pathlib import Path
import base64
import pyheif
import faiss
from facenet_pytorch import MTCNN,InceptionResnetV1
import pickle
import numpy as np
import faiss

class classifier:
    def __init__(self,model_files_path = 'model_files',data_files_path ="data",login = {'host' : "localhost",
    'dbname' : "embeds",
    'user' : "postgres",
    'password' : "282606",
    'port' : 5432},liveness_check = True, liveness_ear_threshold = 0.21, face_detection_threshold = 0.9,USE_GPU = False):
        self.device = torch.device('cuda') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')
        self.data_files_path = Path(data_files_path)
        self.model_files_path = Path(model_files_path)
        self.face_detection_threshold = face_detection_threshold
        self.cosine_confidence = 0.7
        # making connection to db
        try:
            self.conn = psycopg2.connect(**login)
            self.conn.set_session(autocommit=True)
            self.cur = self.conn.cursor()
        except Exception as error:
            print("ERROR: ",error)
            print("ERROR TYPE: ",type(error))

        # instantiating pretrained models
        self.mtcnn = MTCNN(device = self.device)
        self.mtcnn = self.mtcnn.eval()
        self.mtcnn_ = MTCNN(post_process = False,device = self.device)
        self.mtcnn_ = self.mtcnn_.eval()
        self.embedding_generator = InceptionResnetV1(pretrained = 'vggface2').eval().to(self.device)

        #checking if all directories exist
        check_paths = [(self.data_files_path / "train"),(self.model_files_path/"saved_models")]
        for path in check_paths:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

        self.faiss_index = None
        if (self.model_files_path/"saved_models/faissHNSW.index").exists():
            self.faiss_index = faiss.read_index(str(self.model_files_path/"saved_models/faissHNSW.index"))

        #liveness checker - basic blink tester
        if liveness_check:
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(str(self.model_files_path/"saved_models/shape_predictor_68_face_landmarks.dat"))
        
        self.liveness_ear_threshold = liveness_ear_threshold
        

    def removeBatch(self,batch:str,exceptions:str=None):
        pass

    def addData(self, test_img, label):
        pass

    def train(self,batch_size = 1):
        dir_path = self.data_files_path/"train"

        #checking if the directories are in the proper format and/or have images   
        self.num_images = len(list(dir_path.glob("*/*.jpg")))
        if self.num_images == 0:
            return 100 
        self.updateDataFile(batch_size)
    
    def updateDataFile(self,batch_size=32):
        dir_path = self.data_files_path/"train"
        #this will loop for all users batch-wise
        all_users_list = []
        for dir in dir_path.iterdir():
            if dir.is_dir():
                all_users_list.append(str(dir.name))

        imgs_batch = []
        total_added_embeddings = 0
        self.cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings(
                    roll VARCHAR(8),
                    embed BYTEA
                );
            """)       
        x_train = []
        for user in all_users_list:
            img_paths = list((dir_path/user).glob("*"))
            for img_path in img_paths:
                img = self.openImg(img_path)
                imgs_batch.append(img)
                del(img)
                if len(imgs_batch) == batch_size:
                    imgs_batch_tensor = torch.from_numpy(np.stack(imgs_batch))
                    del(imgs_batch[:])
                    embeddings = self.getEmbedding(imgs_batch_tensor)
                    del(imgs_batch_tensor)

                    #updating in file:
                    for embedding in embeddings:
                        self.cur.execute("""
                            insert into embeddings(roll,embed) values (%s,%s)
                        """,(user,pickle.dumps(embedding)))
                        x_train.append(embedding)
                        total_added_embeddings += 1

                    del(embeddings)
                
            #if the user loop is finished i.e. the batch size is more than the total items:
            if len(imgs_batch) > 0:
                imgs_batch_tensor = torch.from_numpy(np.stack(imgs_batch))
                embeddings = self.getEmbedding(imgs_batch_tensor)
                del(imgs_batch[:])
                del(imgs_batch_tensor)
                #updating in file:
                for embedding in embeddings:
                    self.cur.execute("""
                            insert into embeddings(roll,embed) values (%s,%s)
                        """,(user,pickle.dumps(embedding)))
                    x_train.append(embedding)
                    total_added_embeddings += 1

                del(embeddings)
                
            print(f"{user} done.")
        print(f"CREATED {total_added_embeddings} EMBEDDINGS")      
        if self.faiss_index is None:
            self.generateIndex()
        else:
            self.faiss_index.add(np.stack(x_train))
            faiss.write_index(self.faiss_index,str(self.model_files_path/"saved_models/faissHNSW.index"))
        print(f"INDEXED {total_added_embeddings} EMBEDDINGS AND SAVED TO DISK.")

    def generateIndex(self, m=128, efconstruction=64, efsearch=250):
        self.cur.execute("select * from embeddings")
        x_train = [pickle.loads(i[1]) for i in self.cur.fetchall()]
        x_train = np.stack(x_train)

        index_tmp = faiss.IndexHNSWFlat(512,m,faiss.METRIC_INNER_PRODUCT)
        index_tmp.efconstruction = efconstruction
        index_tmp.efsearch = efsearch
        index_tmp.add(x_train)

        self.faiss_index = index_tmp
        del(x_train)
        del(index_tmp)
        faiss.write_index(self.faiss_index,str(self.model_files_path/"saved_models/faissHNSW.index"))

    def del_pretrained(self):
        (self.model_files_path/"saved_models/faissHNSW.index").unlink(missing_ok=True)
        del(self.faiss_index)
        self.faiss_index = None

    def test(self,img_path = None,check_liveness = False,device = 0,num_samples=1,verbose = False,k=20):
        if self.faiss_index == None:
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
                img = self.openImg(img_path)
                if img is not None:
                    imgs_batch.append(img)
            
                
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
        
        if verbose:
            from time import time
            st = time()
        if len(imgs_batch)==0:
            return ""
        imgs_batch_tensor = torch.from_numpy(np.stack(imgs_batch))
        embeddings = self.getEmbedding(imgs_batch_tensor)

        if embeddings is None:
            return ""
        if verbose:
            end = time()
            print(f"processing m {end-st}")
        
            st_search = time()

        distances, labels = self.faiss_index.search(embeddings,k)
        roll_labels = []
        roll_labels_to_score = defaultdict(int)
        for big_i in range(len(distances)):
            for small_i in range(len(labels[big_i])):
                index = int(labels[big_i][small_i])
                self.cur.execute("""
                    select roll from embeddings where embed = %s
                """,(pickle.dumps(self.faiss_index.reconstruct(index)),)
                )
                label = self.cur.fetchall()[0][0]
                print(f"label: {label}  distance: {distances[big_i][small_i]}")
                if distances[big_i][small_i] >= self.cosine_confidence:
                    roll_labels_to_score[label] += distances[big_i][small_i]
        if verbose:
            end_search = time()
            print(f"searching me {end_search-st_search}")
        del(embeddings)
        del(imgs_batch_tensor)

        #saving the last recorded face 
        Path("lastentry").mkdir(parents=True,exist_ok=True)
        self.mtcnn_(imgs_batch[-1],save_path = "lastentry/lastentry.jpg")
        dim = imgs_batch[-1].shape
        del(imgs_batch[:])
        
        if verbose:
            end_ = time()
            print(f"total ekdum return s phle: {end_ - st}")
            print(f"total fps if {num_samples} photo(s) of {dim[0]}x{dim[1]} are given: {num_samples/(end_ - st)}")
        max_key = ""
        if len(roll_labels_to_score)>0:    
            max_key = max(roll_labels_to_score, key=roll_labels_to_score.get)
        return max_key
    
    def videoStream(self,vid_path = None,op_path:str="outputvideo.mp4"):
        if self.faiss_index == None:
            return 106
        try:
            import skvideo.io  
            mtcnn = MTCNN(keep_all=True,device=self.device)
            if vid_path == None:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"fps: {fps}")
            writer = skvideo.io.FFmpegWriter(op_path,inputdict={'-r': str(fps)})
            fontScale = 0.8
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontColor = (255, 255, 255) # white
            thickness = 2
            while(True):
                ret, test_img = cap.read()
                if not ret:
                    break
                test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    boxes, _ = mtcnn.detect(test_img)
                    test_img1 = test_img
                    test_img1 = np.stack(test_img1)
                
                    faces = mtcnn(test_img1)
                    del(test_img1)
                if faces is not None:
                    with torch.no_grad():
                        faces = faces.to(self.device)
                        embeds = self.embedding_generator(faces)
                        del(faces)
                    embeds = embeds.to(torch.device("cpu")).detach().numpy()
                    distances, labels = self.faiss_index.search(embeds,k=5)
                    del(embeds)
                    roll_labels = []
                    roll_labels_to_score = defaultdict(int)
                    for big_i in range(len(distances)):
                        roll_labels_to_score.clear()
                        for small_i in range(len(labels[big_i])):
                            index = int(labels[big_i][small_i])
                            self.cur.execute("""
                                select roll from embeddings where embed = %s
                            """,(pickle.dumps(self.faiss_index.reconstruct(index)),)
                            )
                            label = self.cur.fetchall()[0][0]
                            if distances[big_i][small_i] >= self.cosine_confidence:
                                roll_labels_to_score[label] += distances[big_i][small_i]
                        if len(roll_labels_to_score)>0:    
                            max_key = max(roll_labels_to_score, key=roll_labels_to_score.get)
                            roll_labels.append(max_key)
                        else:
                            roll_labels.append("no data")
                    print(roll_labels)
                    for i,box in enumerate(boxes):
                        x,y,w,h = box.astype(int)
                        cv2.rectangle(test_img,(x,y),(w,h), (0,0,255),2)
                        cv2.putText(test_img, roll_labels[i], (x,h), font, fontScale, fontColor, thickness)
                writer.writeFrame(test_img)
        finally:
            # cv2.waitKey(0)
            writer.close()
            # writer.release()
            cap.release()
            # out.release()
            cv2.destroyAllWindows()


    def openImg(self,test_img:str=None):
        if test_img is None:
            return
        img_path = Path(test_img)
        extension = img_path.suffix.lower()
        if extension == '.heic':
            format = "heic"
        elif extension in ['.jpg', '.jpeg', '.png']:
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
            height, width, channels = test_img.shape
            aspect_ratio = width / height
            if(width>1000):
                new_width = 720
                new_height = int(new_width / aspect_ratio)
                test_img = cv2.resize(test_img,(new_width,new_height))
            #enforcing resize to 640x480
            test_img = cv2.resize(test_img,(640,480))
            
            return test_img
        except Exception as error:
            print("could not open image ",img_path)
            print("ERROR: ",error)
            print("error type: ",type(error))
            return None

    def getEmbedding(self,test_img, numpy = True):    #accepting
        try:
            with torch.no_grad():
                faces = self.mtcnn(test_img)
                del(test_img)
                faces = [i for i in faces if i is not None]
                faces = torch.stack(faces).to(self.device)
                embeddings = self.embedding_generator(faces)
                del(faces)
                if numpy:
                    embeddings = embeddings.to(torch.device("cpu")).detach().numpy()
        except Exception as error:
            print("error: ",error)
            print("type: ",type(error))
            embeddings = None #no face found
        return embeddings
 
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
        try:
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
        except:
            return None, None

    def __del__(self):
        self.cur.close()
        self.conn.close()