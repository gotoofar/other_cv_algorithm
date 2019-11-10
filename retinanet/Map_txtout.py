


import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import csv

'''this program is for map_calcu
   input : (image folder location , out txt or csv location,the model used)
   output: a txt  indicate the annovation and class of image in folder
           format (XXX.jpg confi,x1,y1,x2,y2
                   XXX.jpg confi,x1,y1,x2,y2
                   (one rectangle a line)
                    ...
                    ...)

'''
import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms


from val_dataloader import Mycollater,MyResizer,MyDataset,MyNormalizer,MyUnNormalizer

colors = {'000': (0, 0, 255), '001': (0, 255, 0), '002': (255, 0, 0), '003': (0, 255, 255), '004': (255, 0, 255),
          '005': (255, 255, 0), '006': (255, 255, 255), '007': (0, 127, 127), '008': (127, 0, 255)}



assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))



def Detect(csv_val,csv_classes,model,save_path,out_txt):

    dataset_val = MyDataset(val_file=csv_val, class_list=csv_classes,
                                 transform=transforms.Compose([MyNormalizer(), MyResizer()]), val=True)


    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=Mycollater, batch_sampler=None)

    retinanet = torch.load(model)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    unnormalize = MyUnNormalizer()

    #f = open(out_txt, "w")

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():

            filename = os.path.split(data['name'][0])
            # print(filename[1])
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            scale = data['scale'][0]
            print('cost time: {} second'.format(time.time() - st))
            idxs = np.where(scores > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # rowlist=[]
            # rowlist.append(csv_val + '/' + filename[1])

            out_txtname=out_txt+'/'+filename[1][:-4]+'.txt'
            #print(filename[1])

            for j in range(idxs[0].shape[0]):

                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                scores_confi=scores[idxs[0][j]]




                with open(out_txtname, 'a+') as f:
                    f.write(label_name+' '+str(scores_confi.cpu().numpy()))
                    f.write(' ' +str(x1/scale)+' '+str(y1/scale)+' '+str(x2/scale)+' '+str(y2/scale))
                    f.write('\n')

                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=colors[label_name], thickness=2)

            #writer.writerow(rowlist)
            # print(label_name)
            #cv2.imwrite(save_path + '/' + filename[1],img)
            cv2.imshow('img', img)
            cv2.waitKey(1)


if __name__ == '__main__':

    model='/home/cbird/work/net/retinanet/csv_retinanetbig_91.pt'
    valimg_folder='/home/cbird/work/data/data_big2/val'
    csv_class='/home/cbird/work/data/data_big2/class_csv'
    out_csv = '/home/cbird/work/data/data_big2/result_csv'
    out_txt='/home/cbird/work/net/Map_calcu/retinapre'
    save_path = "/home/cbird/work/data/data_big2/valresult"        # if you wanna save the img with rectangle

    if os.path.exists(out_txt) != True:
        os.makedirs(out_txt)

    Detect(valimg_folder,csv_class,model,save_path,out_txt)

