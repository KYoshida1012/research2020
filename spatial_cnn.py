import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
import subprocess

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image

import dataloader
from utils import *
from network import *
#from plot import *
from chart_studio.plotly import *
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs')
#parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 25)')
#parser.add_argument('--batch-size', default=25, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--selection', default='', type=str, metavar='N', help='image selection')

def main():
    global arg
    arg = parser.parse_args()
    print(arg)

    data_path           = './datasets/'
    video_list_path     = 'gaze_list'
    split               = '01'
    pretrain_weights    = 'pretrained_models/spatial_model_best.pth.tar'
    gazemap_size        = 56

    extract_frames(data_path, video_list_path, split)

    #Prepare DataLoader
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path=data_path,
                        video_list_path=video_list_path,
                        split=split, 
                        gazemap_size=gazemap_size,
                        )
    
    train_loader, test_loader, selection_loader = data_loader.run()
    print ('Traing Data Length: {0}'.format(len(train_loader)))
    print ('Test Data Length: {0}'.format(len(test_loader)))

    #Model 
    model = Spatial_CNN(
                        nb_epochs           =arg.epochs,
                        lr                  =arg.lr,
                        batch_size          =arg.batch_size,
                        resume              =arg.resume,
                        start_epoch         =arg.start_epoch,
                        evaluate            =arg.evaluate,
                        selection           =arg.selection,
                        train_loader        =train_loader,
                        test_loader         =test_loader,
                        selection_loader    =selection_loader,
                        pretrain_weights    =pretrain_weights,
                        gazemap_size        =gazemap_size
                        # test_video=test_video
    )
    #Training
    model.run()
    #model.resume_and_evaluate(self.resume=true);

def extract_frames(data_path, video_list_path, split):
    frame_path = os.path.join(data_path, 'frame')
    train_list_file = os.path.join(video_list_path, 'trainlist'+split+'.txt')
    test_list_file = os.path.join(video_list_path, 'testlist'+split+'.txt')

    # read train video lists into a train_videos
    with open(train_list_file) as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
    f.close()
    
    train_videos = []
    for line in content:
        # every line is a video's name
        train_videos.append(line.split(' ',1)[0])

    # read train video lists into a train_videos
    with open(test_list_file) as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
    f.close()
    
    test_videos = []
    for line in content:
        # every line is a video's name
        test_videos.append(line.split(' ',1)[0])

    # extract train frames
    print('extracting train frames...')
    for video in train_videos:
        print(video)
        video_path = os.path.join(data_path, 'video', video)
        train_frame_path = os.path.join(frame_path, video.split('.')[0])
        print(train_frame_path)
        if not os.path.isdir(train_frame_path):
            os.mkdir(train_frame_path)
            print('ffmpeg -i {} -s 448x448 -r 30 {}/frame%06d.jpg'.format(video_path, train_frame_path))
            subprocess.call('ffmpeg -i {} -s 448x448 -r 30 {}/frame%06d.jpg'.format(video_path, train_frame_path), shell=True)

    # extract test frames
    print('extracting test frames...')
    for video in test_videos:
        print(video)
        video_path = os.path.join(data_path, 'video', video)
        test_frame_path = os.path.join(frame_path, video.split('.')[0])
        if not os.path.isdir(test_frame_path):
            os.mkdir(test_frame_path)
            subprocess.call('ffmpeg -i {} -s 448x448 -r 30 {}/frame%06d.jpg'.format(video_path, test_frame_path), shell=True)


class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, selection, train_loader, test_loader, selection_loader, pretrain_weights, gazemap_size):
        self.nb_epochs          =   nb_epochs
        self.lr                 =   lr
        self.batch_size         =   batch_size
        self.resume             =   resume
        self.start_epoch        =   start_epoch
        self.evaluate           =   evaluate
        self.selection          =   selection
        self.train_loader       =   train_loader
        self.test_loader        =   test_loader
        self.selection_loader   =   selection_loader
        # self.best_prec1       =   0
        self.min_loss           =   0
        self.pretrain_weights   =   pretrain_weights
        self.gazemap_size       =   gazemap_size
        # self.test_video       =   test_video
        self.logger_path        =   'record/logger.json'
        self.logger             =   {}
        self.logger['train']    =   {}
        self.logger['val']      =   {}
        self.visu_path          =   'record/gaze_visu.html'
        self.output_path        =   'record/output'
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=3, gazemap_size=self.gazemap_size).cuda()
        # self.model = resnet101(pretrained= True, channel=3).cuda()
        #Loss function and optimizer
        self.criterion = nn.MSELoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            # resume from pretrained 2-stream CNN weight
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                print(checkpoint['state_dict'].keys())
                self.start_epoch = checkpoint['epoch']
                # self.best_prec1 = checkpoint['best_prec1']
                self.min_loss = checkpoint['min_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(self.resume, checkpoint['epoch'], self.best_prec1))
                print("==> loaded checkpoint '{}' (epoch {}) (min_loss {})".format(self.resume, checkpoint['epoch'], self.min_loss))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))

        if self.evaluate:
            self.epoch = 0
            # prec1, val_loss = self.validate_1epoch()
            val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True

        if self.selection:
            print ('selection process...')
            self.selection_process()

        else:
            self.epoch = 0
            #val_loss = self.validate_1epoch()
            val_loss = self.train_1epoch()
            with open(self.logger_path, 'w') as f:
                json.dump(self.logger, f)

            # visualization
            visu_one_exp(self.logger_path, self.visu_path)

            for self.epoch in range(self.start_epoch+1, self.nb_epochs+1):
                self.train_1epoch()
                val_loss = self.validate_1epoch()
                # is_best = prec1 > self.best_prec1
                is_best = val_loss < self.min_loss

                #lr_scheduler
                self.scheduler.step(val_loss)

                # save model
                if is_best:
                    self.min_loss = val_loss
                    # with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    #     pickle.dump(self.dic_video_level_preds,f)
                    # f.close()

                save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    # 'best_prec1': self.best_prec1,
                    'min_loss': self.min_loss,
                    'optimizer' : self.optimizer.state_dict()
                },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

                # save logger for every epoch
                with open(self.logger_path, 'w') as f:
                    json.dump(self.logger, f)

                # visualization
                visu_one_exp(self.logger_path, self.visu_path)


    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train() 
        end = time.time()
        # mini-batch training
        #progress = tqdm(self.train_loader)

        #for i, data in enumerate(progress):
        #for i, data in enumerate(self.train_loader):
        
        for data in tqdm(self.train_loader):
            video_name, img, gaze, = data

            # measure data loading time
            data_time.update(time.time() - end)
            
            img = img.cuda()
            target = gaze.type(torch.FloatTensor).cuda()
            target_var = Variable(target).cuda()

            model = self.model.cuda()
            output = model(img)

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            losses.update(loss.cpu().item(), img.cpu().size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')
        self.logger['train'][str(self.epoch)] = losses.avg

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        # self.dic_video_n_frame = {}
        # self.dic_video_level_losses = {}
        loss_sum = 0
        count = 0
        end = time.time()
        progress = tqdm(self.test_loader)
        #print(progress)

        for i, data in enumerate(progress):
            # if (i>50):
            #     break
            # for i, data in enumerate(self.train_loader):
            #print ("loop")
            video_name, img, gaze = data

            data_time.update(time.time() - end)
            
            img = img.cuda()
            target = gaze.type(torch.FloatTensor).cuda()
            target_var = Variable(target).cuda()

            model = self.model.cuda()
            output = model(img)
            loss = self.criterion(output, target_var)
            # print('loss:',loss.size(), loss)

            # measure accuracy and record loss
            losses.update(loss.cpu().item(), img.cpu().size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i==100 or i==234 or i == 300 or i == 400 or i==500 or i ==600 or i == 700 or i == 800 or i == 900 or i == 1000 or i == 1100 or i == 1200 or i == 1300 or i == 1400 or i == 1500 or i == 1600 or i == 1700 or i== 1800 or i == 1900 or i == 2000:
                pic = torch.cat((target.data.cpu().view(1,1,self.gazemap_size,self.gazemap_size), output.data.cpu().view(1,1,self.gazemap_size,self.gazemap_size)), 0)
                #print(pic)
                save_image(pic, os.path.join(self.output_path, 'batch{}_epoch{}.jpg'.format(i, self.epoch)), nrow = 8)

            # # visualize predicted gaze map
            # if (i < 1300 and i > 1250):

            #     # print(target.data.cpu().size(), output.data.cpu().size())
            #     pic = torch.cat((target.data.cpu().view(1,1,14,14), output.data.cpu().view(1,1,14,14)), 0)
            #     # print(pic.shape, pic)
            #     save_image(pic, os.path.join(self.output_path, 'batch{}_epoch{}.jpg'.format(i, self.epoch)), nrow = 8)


            nb_data = img.cpu().size(0)
            loss_sum += loss.cpu().item() * nb_data
            count += nb_data

        loss_avg = loss_sum / count

            # #Calculate video level prediction
            # preds = output.data.cpu().numpy()
            # nb_data = loss.shape[0]
            # for j in range(nb_data):
            #     # videoName = keys[j].split('/',1)[0]
            #     videoName = video_name[j]
            #     if videoName not in self.dic_video_level_losses.keys():
            #         self.dic_video_level_losses[videoName] = loss[j,:]
            #         self.dic_video_n_frame[videoName] = 1
            #     else:
            #         self.dic_video_level_losses[videoName] += loss[j,:]
            #         self.dic_video_n_frame[videoName] += 1

        # video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
        # video_loss = self.frame2_video_level_accuracy()

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(losses.avg,5)]}
                # 'Prec@1':[round(video_top1,3)],
                # 'Prec@5':[round(video_top5,3)]}
        record_info(info, 'record/spatial/rgb_test.csv','test')
        self.logger['val'][str(self.epoch)] = losses.avg
        # return video_top1, video_loss
        # return video_loss
        return loss_avg

    def selection_process(self):
        print('Selection Process')
        batch_time      = AverageMeter()
        losses          = AverageMeter()
        data_time       = AverageMeter()
        top1            = AverageMeter()
        top5            = AverageMeter()
        self.model.eval()
        loss_sum        = 0
        count           = 0
        end             = time.time()
        progress        = tqdm(self.selection_loader)

        topcount_sum        = np.array([0, 0, 0, 0])
        topcount_max        = np.array([0, 0, 0, 0])
        for i, data in enumerate(progress):
            imgs, video_index = data

            outputs = []
            for img in imgs:
                img         = img.cuda()
                model       = self.model.cuda()
                outputs.append( model(img) )
                #save_image(pic, os.path.join(self.output_path, 'batch{}_epoch{}.jpg'.format(i, self.epoch)), nrow = 8)

            sums = self.sum( outputs )
            maxs = self.max( outputs )
            selected_index = sums[0][0]
            topcount_sum = topcount_sum + self.topcounter(video_index, sums)
            topcount_max = topcount_max + self.topcounter(video_index, maxs)
            print ('selected-index, ', selected_index, ', recorded-index, ',  video_index)
            count = count + 1
        loss_avg        = 0
        #loss_sum / count
        print(topcount_sum)
        print(topcount_max)
        topcount_sum = topcount_sum / count
        topcount_max = topcount_max / count
        print(topcount_sum)
        print(topcount_max)

#        info = {'Epoch': [self.epoch],
#                'Batch Time': [round(batch_time.avg, 3)]
#                'Loss': [round(losses.avg, 5)]}
#        record_info(info, 'record/spatial/rgb_test.csv', 'test')
#        self.logger['val'][str(self.epoch)] = losses.avg
        return loss_avg

    def sum(self, inputs):
        sums    = []
        for index, input in enumerate(inputs):
            sums.append( (index+1, torch.sum( input ).cpu().item()))
        sums.sort(key = secondVal, reverse = True)
        print(sums)
        return sums

    def max(self, inputs):
        maxs = []
        for index, input in enumerate(inputs):
            maxs.append((index + 1, torch.max(input).cpu().item()))
        maxs.sort(key=secondVal, reverse=True)
        print(maxs)
        return maxs

    def topcounter(self, truth, inputs):
        for index, input in enumerate(inputs):
            if(truth == input[0]):
                rank = index + 1
        counts = np.array([])
        for index, input in enumerate(inputs):
            if(index+1 >= rank):
                counts = np.append(counts, np.array([1]))
            else:
                counts = np.append(counts, np.array([0]))
        return counts

def secondVal(val):
    return val[1]

if __name__=='__main__':
    main()
