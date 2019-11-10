


import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import csv

'''this program is for calling only
   input : (image folder location , out txt or csv location,the model used)
   output: a txt or csv indicate the annovation and class of image in folder
           format (XXX.jpg x1,y1,x2,y2,classname,x1,y1,x2,y2,classname...
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



def Detect(csv_val,csv_classes,model,save_path,out_csv):

    dataset_val = MyDataset(val_file=csv_val, class_list=csv_classes,
                                 transform=transforms.Compose([MyNormalizer(), MyResizer()]), val=True)


    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=Mycollater, batch_sampler=None)

    retinanet = torch.load(model)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    unnormalize = MyUnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    csv_file = open(out_csv, 'w', newline='')   # XX.jpg,x1,y1,x2,y2,class,x1,y1,x2,y2,class...
    writer = csv.writer(csv_file)
    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():

            filename = os.path.split(data['name'][0])
            scale=data['scale'][0]

            # print(filename[1])
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print('cost time: {} second'.format(time.time() - st))
            idxs = np.where(scores > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            rowlist=[]
            rowlist.append(csv_val + '/' + filename[1])

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                rowlist.extend([x1/scale,y1/scale,x2/scale,y2/scale,label_name])    # divide scale can turn image scale to before format
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=colors[label_name], thickness=2)
            writer.writerow(rowlist)
            # print(label_name)
            #cv2.imwrite(save_path + '/' + filename[1],img)
            cv2.imshow('img', img)
            cv2.waitKey(1)


if __name__ == '__main__':

    model='/home/cbird/work/net/retinanet/csv_retinanetbig_91.pt'
    valimg_folder='/home/cbird/work/data/data_big2/val'
    csv_class='/home/cbird/work/data/data_big2/class_csv'
    out_csv = '/home/cbird/work/data/data_big2/result_csv'
    save_path = "/home/cbird/work/data/data_big2/valresult"        # if you wanna save the img with rectangle

    if os.path.exists(save_path) != True:
        os.makedirs(save_path)

    Detect(valimg_folder,csv_class,model,save_path,out_csv)

