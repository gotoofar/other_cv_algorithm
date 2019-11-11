# -*- coding:utf-8 -*-
import torch

import sys
import os
import csv

# 把pascal格式的xml数据转换成这种容易处理的 txt


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


#xml转txt
def totxt():
    xml_path='G:/git_folder/retinanet/data_big/Annotations'   #文件夹位置
    #image_sets = 'G:/git_folder/retinanet/data_big/ImageSet/Main/trainval.txt'    # 训练样本集
    out_train_txt='G:/git_folder/retinanet/data_big/train.txt'

    f = open(out_train_txt, "w")
    xml_list = os.listdir(xml_path)
    for xml_name in xml_list:
        print('xml'+xml_name)
        sub_xmlname = os.path.join(xml_path, xml_name)
        target = ET.parse(sub_xmlname).getroot()
        f.write(xml_name[:-3]+'jpg')

        for obj in target.iter('object'):
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = bbox.find(pt).text
                f.write(' '+cur_pt)
                bndbox.append(cur_pt)
                #写入文件
            classname=obj.find('name').text
            f.write(' '+classname)
        f.write('\n')


            #写入文件



#xml文件转为retinanet可用的csv文件

# 顺便把class_csv也生成

def tocsv():
    #标注文件位置
    xml_file='G:/git_folder/retinanet/data_big/Annotations'
    #输出的csv位置
    out_csv='G:/git_folder/retinanet/data_big/test_csv'
    class_csv='G:/git_folder/retinanet/data_big/classtest_csv'

    #图像的位置
    img_path='/home/cbird/work/data/data_big2/val/'

    csv_file=open(out_csv,'w',newline='')
    class_file=open(class_csv,'w',newline='')
    writer=csv.writer(csv_file)
    class_writer=csv.writer(class_file)
    xml_list=os.listdir(xml_file)


#class_csv里面需要放的东西  这里写得有点笨了
    class_writer.writerow(['000','0'])
    class_writer.writerow(['001','1'])
    class_writer.writerow(['002','2'])
    class_writer.writerow(['003','3'])
    class_writer.writerow(['004','4'])
    class_writer.writerow(['005','5'])
    class_writer.writerow(['006','6'])
    class_writer.writerow(['007','7'])
    class_writer.writerow(['008','8'])




    for xml in xml_list:
        sub_xmlname = os.path.join(xml_file, xml)
        target = ET.parse(sub_xmlname).getroot()
        for obj in target.iter('object'):
            bbox=obj.find('bndbox')
            pts=['xmin','ymin','xmax','ymax']
            bndbox=[]
            for i,pt in enumerate(pts):
                cur_pt=bbox.find(pt).text
                bndbox.append(cur_pt)
            classname = obj.find('name').text
            writer.writerow([img_path+xml[:-3] + 'jpg',bndbox[0],bndbox[1],bndbox[2],bndbox[3],classname])



if __name__ == '__main__':
    totxt()



