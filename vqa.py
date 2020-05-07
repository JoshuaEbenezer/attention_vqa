#from torch_lr_finder import LRFinder
import cv2
import os
from PIL import Image
import copy
import scipy.stats
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import numpy as np
from paq2piq.inference_model import *
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from torchvision import transforms, utils,models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./logs')

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1,x.size(-3),x.size(-2), x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(0), y.size(-1))  # (timesteps, samples, output_size)

        return y

class VQAModel(nn.Module):
    def __init__(self,nframes):
        super(VQAModel,self).__init__()
        self.base_model_class = InferenceModel(RoIPoolModel(), 'RoIPoolModel.pth')
        base_model = self.base_model_class.model
        self.iqa_model = nn.Sequential(base_model.body,base_model.head[0:2])
        self.iqa_fc = nn.Sequential(nn.Linear(1024,16),nn.ReLU())
#        self.fd_model_class = InferenceModel(RoIPoolModel(), 'RoIPoolModel.pth')
#        fd_model = self.fd_model_class.model
#        self.frame_diff_model = nn.Sequential(fd_model.body,fd_model.head[0:2],nn.Linear(1024,16),nn.ReLU())
#        iqa_model = base_model_class.model
##        iqa_model =nn.Sequential(base_model.body,base_model.head[0:2])
##        iqa_model = iqa_model.to(device)
        for param in self.iqa_model.parameters():
            param.requires_grad = False
        self.T1 = TimeDistributed(nn.Sequential(self.iqa_model,self.iqa_fc))
 #       self.T2 = TimeDistributed(self.frame_diff_model)
#        self.rnn1 = nn.LSTM(1024,512) 
#        self.rnn2 = nn.LSTM(1024,512) 
        self.multihead1 = nn.MultiheadAttention(128,4,dropout=0.1)
  #      self.multihead2 = nn.MultiheadAttention(16,1)
        self.fc = nn.Sequential(nn.Linear(128,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,1))#,nn.ReLU(),nn.Dropout(0.02),nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.04),nn.Linear(32,1))
#        self.fc1 = nn.Sequential(nn.Linear(1024,128),nn.LeakyReLU(),nn.Linea64128,1)) 
    def forward(self,video):
        #        video = torch.squeeze(video)
        #        self.iqa_model.input_block_rois(self.blk_size, [video.shape[-2], video.shape[-1]], device=device)
        #        t = []
        #        for i in range(video.shape[0]):
        #            image = torch.unsqueeze(video[i,:,:,:],0)
        #            score = self.iqa_model(image).data[0]
        #            t.append(torch.unsqueeze(score[0],0))
        #        t = torch.cat(t)
        features = self.T1(video)
#        fd_features = self.T2(frame_diff)
#        self.rnn2.flatten_parameters()
#        recurrent_fd_features,_ = self.rnn2(fd_features)
#
#        self.rnn1.flatten_parameters()
#        recurrent_features,_ = self.rnn1(features)
        attention,_ = self.multihead1(features,features,features)
#        attention_fd,_ = self.multihead2(fd_features,fd_features,fd_features)
        attention_mean = torch.mean(attention,0)
#        attention_mean_fd = torch.mean(attention_fd,0)
 #       attention_fc = torch.cat((attention_mean,attention_mean_fd),1)
        output = self.fc(attention_mean)
  #      frame_diff_features = self.T2(frame_diff)
   #     features =torch.cat((frame_features,frame_diff_features),2)
 #       attention,_ = self.multihead(features,features,features)
#        attention = torch.reshape(attention,(1,15))
 #       output =torch.mean(self.fc(attention)).view(1,1)# self.fc(attention)
#        attention_mean = torch.mean(attention,0)
#        output = attention_mean#orch.unsqueeze(attention_mean,0) 
#        output = self.fc1(attention_mean)
        return output
class KonvidDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vid_name = os.path.join(self.root_dir,
                                    self.landmarks_frame.iloc[idx, 0])
        vid_frames = self.landmarks_frame.iloc[idx,1]
        score = np.asarray(self.landmarks_frame.iloc[idx,2])
        score = score.astype(np.float32)/4.64*100
        frames_to_read =np.linspace(1,vid_frames-5,10,dtype=np.int16)
#        rand_add = np.random.randint(1,5,(10,))
 #       frames_to_read = frames_to_read+rand_add
        cap = cv2.VideoCapture(vid_name)
#        video = np.zeros((15,540,960,3))
        video = []
        frame_diff=[]
        index=0
        count=0
        while(True):
            ret,frame = cap.read()
            if(ret==False):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)/255.0
#            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            if(count==frames_to_read[index]):
                video.append(torch.unsqueeze(frame,0))
                #                video[index,:,:,:] = frame
                index=index+1
                if(index==10):
                    break
            count = count+1
            prev = frame
        videoTensor = torch.Tensor(10,3,540,960)
        torch.cat(video,out=videoTensor)
        sample = {'video': videoTensor, 'mos': torch.from_numpy(np.expand_dims(score,axis=0)).float()}
        return sample
def train_model(model, criterion, optimizer,  dataloaders,num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            preds = []
            scores = []
            running_loss = 0.0
            databar = tqdm(dataloaders[phase])
            # Iterate over data.
            for sample_batched in databar:

                video_batch,score = sample_batched['video'],sample_batched['mos']

                score = score.to(device,dtype=torch.float)
                video_batch = video_batch.to(device,dtype=torch.float)
#                frame_diff = frame_diff.to(device,dtype=torch.float)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    pred_mos = model(video_batch)
                    loss = criterion(pred_mos, score)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if(phase=='val'):
                        preds.append(pred_mos.detach().cpu().numpy())
                        scores.append(score.detach().cpu().numpy())
                databar.set_description("Current loss %f" % loss.item())
                # statistics
                running_loss += loss.item() * video_batch.size(0)
#            if phase == 'train':
                #                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            print(phase,' running loss is :',  epoch_loss)
            # deep copy the model

            if phase == 'val': 
                torch.save(copy.deepcopy(model.state_dict()), 'att_latest_model.pth')
                preds = np.asarray(preds)
                scores = np.asarray(scores)
                srocc = scipy.stats.spearmanr(preds.flatten(),scores.flatten())
                print('SROCC is ', srocc)
                if(epoch==0):
                    best_srocc = srocc[0]
                writer.add_scalar('Val loss',epoch_loss)
                writer.add_scalar('Val SROCC', srocc[0])
                if srocc[0] >= best_srocc:
                    best_srocc = srocc[0]
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'att_best_model.pth')
                    print("Saving better model")
            else:
                writer.add_scalar('Train loss',epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
model = VQAModel(nframes=10)
model = model.to(device)
model = nn.parallel.DataParallel(model)
train= True
transform =transforms.ToTensor()#transforms.Compose([transforms.CenterCrop((640,640)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = KonvidDataset('konvid_names.csv','/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/konvid/KoNViD_1k_videos/',transform)
print(len(dataset))

train_size =int(0.8*len(dataset))
val_size =int(0.1*len(dataset))
test_size = len(dataset)-train_size-val_size
train_set, val_set,test_set = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_loader = DataLoader(train_set, batch_size=3,shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=3,shuffle=True, num_workers=4)
dataloaders = {"train":train_loader,"val":val_loader}

if(train==True):
    criterion = nn.MSELoss()
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    # Observe that (not) all parameters are being optimized
    
    optimizer = optim.Adam(params_to_update,lr=1e-4,weight_decay=0.5)#[{'params':model.module.iqa_fc.parameters(),'lr':1e-4},{'params':model.module.fc.parameters(),'lr':1e-4},{'params':model.module.multihead1.parameters(),'lr':1e-4}],weight_decay=0.005)
    clr_scheduler =torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9) 
#    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
#    lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=0.001, num_iter=100, step_mode="linear")
#    lr_finder.plot(log_lr=False)
#    lr_finder.reset()

    # LR by a factor of 0.1 every 7 epochs
    model_conv = train_model(model, criterion, optimizer,dataloaders, num_epochs=50)
else:
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False,num_workers=8)
    PATH = 'best_model.pth'
    model.load_state_dict(torch.load(PATH))

#    model = InferenceModel(RoIPoolModel(),'RoIPoolModel.pth')
    def predict(model,video):
        print(torch.max(video))
        vid_pred = []
        for i in range(video.shape[0]):
            frame_pred = []
            for j in range(video.shape[1]):
                image = video[i,j,:,:,:]
                frame_pred.append(model.predict(image))
            print(torch.mean(torch.tensor(frame_pred)))
            vid_pred.append(torch.mean(torch.tensor(frame_pred)))
        return torch.Tensor(vid_pred)

#    model.to(device)
#    model.eval()
    sroccs = []
    preds = []
    scores = []
    for sample_batched in tqdm(test_loader):
        video,mos = sample_batched['video'],sample_batched['mos']
        
#        pred = predict(model,video)
        pred = model(video)
        pred = pred.detach().cpu()
        print(pred,mos)
        preds.append(pred)
        scores.append(mos.detach().cpu())
    preds = np.concatenate(preds).ravel()
    scores = np.concatenate(scores).ravel()
    srocc = scipy.stats.spearmanr(preds,scores)
    print(srocc)
    
