# retinanet

> 这个retinanet是记录怎么配置运行大佬的[pytorch-retinanet原链接](https://github.com/yhenon/pytorch-retinanet) 所以之前的README也保留住

按照原README说明把包安装完之后，准备一下三个csv
anno_csv
val_csv
class_csv
前面两个文件里面放训练和验证集的标注数据，形式如
'''
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
'''

class_csv里面是类别名与索引数字
'''
cow,0
cat,1
bird,2
'''

如果需要预训练，可以先下好模型，再到model.py里面修改默认的加载路径
例如
'model.load_state_dict(torch.load('/home/cbird/work/resnet50-19c8e357.pth'),strict=False)'
训练好的模型保存路径也可以传入，如果未传入则保存在当前文件夹下

### train
数据路径在训练时通过参数传入
'''
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
'''

### test
官方的测试是 visualize.py
'python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>'

因为训练和测试时对数据的处理有差异，所以修改了dataloader写了一个val_dataloader，具体在Detectron.py和Map_txtout.py里有引用
因为用pycharm比较习惯，就写了两个可以直接run的，分别保存csv和txt的测试程序，路径什么的都在程序里面修改
Detection.py   保存csv结果以及测试结果图片
Map_txtout.py  保存txt结果以及测试结果图片（Map的计算用这个格式的txt，所以专门写的）
仅仅是方便，没有什么扩展
