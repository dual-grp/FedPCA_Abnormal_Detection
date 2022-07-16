import torch
import os

from FLAlgorithms.users.userADMM import UserADMM
from FLAlgorithms.users.userADMM2 import UserADMM2
from FLAlgorithms.servers.serverbase2 import Server2
# from utils.model_utils import read_data, read_user_data
from utils.test_utils import kdd_test
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

''' Implementation for FedPCA Server'''

class AbnormalDetection(Server2):
    def __init__(self, algorithm, experiment, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time):
        super().__init__(device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time)

        # Initialize data for all  users
        self.algorithm = algorithm
        self.K = 0
        self.experiment = experiment
        dataX = self.get_data_kdd_80000()
        self.num_clients = 20
        factor = 80000/self.num_clients
        self.learning_rate = learning_rate
        self.user_fraction = num_users
        total_users = self.num_clients
        print("total users: ", total_users)
        for i in range(self.num_clients):            
            id = i
            train = self.get_client_data(dataX, factor=factor, i=i)
            train = torch.Tensor(train)
            if(i == 0):
                U, S, V = torch.svd(train)
                V = V[:, :dim]
                # self.commonPCAz = V
                print("type of V", type(V))
                print("shape of V: ", V.size())
                self.commonPCAz = torch.rand_like(V, dtype=torch.float)
                print(self.commonPCAz)
                print(f"Shape of V: {V.shape}")
                check = torch.matmul(V.T,V)

            #user = UserADMM(device, id, train, test, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            # user = UserADMM2(device, id, train, test, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            user = UserADMM2(algorithm, device, id, train, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", int(num_users*total_users), " / " ,total_users)
        print("Finished creating FedPCA server.")


    '''
    Get data from csv file
    '''
    def get_data(self, i):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"client{i+1}_preprocessed.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read preprocessed data from csv file
        client_train = pd.read_csv(client_path)
        client_train = client_train.to_numpy()

        return client_train

    '''
    Get data from kdd dataset csv file
    '''
    def get_data_kdd(self, i):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"client{i+1}_kdd_std.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read preprocessed data from csv file
        client_train = pd.read_csv(client_path)
        client_train = client_train.to_numpy()

        return client_train
    
    '''
    Get 80000 data from kdd dataset csv file
    '''
    def get_data_kdd_80000(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"kdd_80000_34_fea.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read data from csv file
        client_train = pd.read_csv(client_path)

        return client_train

    '''
    Preprocessing data step
    '''
    def prep_data(self, dataX):
        change_dataX = dataX.copy()
        featuresToScale = change_dataX.columns
        sX = StandardScaler(copy=True)
        change_dataX.loc[:,featuresToScale] = sX.fit_transform(change_dataX[featuresToScale])
        return change_dataX

    '''
    Divide data to clients
    '''
    def get_client_data(self, data, factor, i):
        # Read data frame for each client
        factor = int(factor)
        dataX = data[factor*i:factor*(i+1)].copy()
        # Preprocess data
        client_data = self.prep_data(dataX)
        client_data = client_data.to_numpy()
        return client_data
    

    '''
    Training model
    '''
    def train(self):
        current_loss = 0
        prev_loss = 0
        acc_score = 0
        losses_to_file = []
        acc_score_to_file = []
        acc_score_to_file.append(acc_score) # Initialize accuracy as zero
        self.selected_users = self.select_users(1000,1)
        print("All user in the network: ")
        for user in self.selected_users:
            print("user_id: ", user.id)

        # Start estimating wall-clock time
        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")

            self.send_pca()

            # Evaluate model each interation
            prev_loss = current_loss
            current_loss = self.evaluate()
            current_loss = current_loss.item()
            losses_to_file.append(current_loss)

            self.selected_users = self.select_users(glob_iter, self.user_fraction)
            # self.users = self.selected_users 

            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                user.train(self.local_epochs)
                print(f" selected user for training: {user.id}")
            # self.users[0].train(self.local_epochs)
            self.aggregate_pca()

            # Check loss to early terminate training process
            # if abs(prev_loss-current_loss) < 1e-2:
            #     break
            
            # Evaluate the accuracy score
            # Extract common representation
            Z = self.commonPCAz.detach().numpy()
            acc_score = kdd_test(Z, thres_hold=6)
            acc_score_to_file.append(acc_score)

        # End estimating wall-clock time
        end_time = time.time()

        # Extract common representation
        Z = self.commonPCAz.detach().numpy()
        
        # Extract losses to file
        losses_to_file = np.array(losses_to_file)

        # Extract accuracy score to file
        acc_score_to_file = np.array(acc_score_to_file)

        # Save common representation and losses to files
        # Get data path
        if self.algorithm == "FedPG":
            space = "Grassman"
        elif self.algorithm == "FedPE":
            space = "Euclidean"
        directory = os.getcwd()
        data_path = os.path.join(directory, "results/KDD")
        acc_path = os.path.join(data_path, "KDD_acc")
        acc_file_name = f'{space}_acc_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}'
        acc_file_path = os.path.join(acc_path, acc_file_name)
        losses_path = os.path.join(data_path, "KDD_losses")
        losses_file_name = f"{space}_losses_KDD_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}"
        losses_file_path = os.path.join(losses_path, losses_file_name)
        # Store accuracy score to file
        np.save(acc_file_path, acc_score_to_file)
        np.save(losses_file_path, losses_to_file)
        np.save(f'{space}_Abnormaldetection_KDD_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}', Z)
        # np.save(f"Grassman_losses_KDD_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}", losses_to_file)
        print(f"training time: {end_time - start_time} seconds")
        kdd_test(Z, thres_hold=6)
        print("Completed training!!!")
        # self.save_results()
        # self.save_model()