import torch
import torch.nn as nn
import yaml
import math

def make_divisible(x,divisor):
    return math.ceil(x/divisor)*divisor

def autopad(kernel,padding=None):
    if padding is None:
        padding = kernel//2

    return padding

class Conv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=1,stride=1,padding=None,groups=1,activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding=autopad(kernel_size,padding),groups=groups,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.1,inplace=True) if activation else nn.Identity()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,shortcut=True,groups=1,expansion=0.5):
        super().__init__()
        hidden_channel = int(out_channel*expansion)
        self.conv1 = Conv(in_channel,hidden_channel,1,1)
        self.conv2 = Conv(hidden_channel,out_channel,3,1,groups= groups)
        self.add = shortcut and in_channel==out_channel 
    def forward(self,x):
        if self.add:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))

class BottleneckCSP(nn.Module):
    def __init__(self,in_channel,out_channel,repeats=1,shortcut=True,groups=1,expansion=0.5):
        super().__init__()
        hidden_channel = int(out_channel*expansion)
        self.conv1 = Conv(in_channel,hidden_channel,1,1)
        self.conv2 = nn.Conv2d(in_channel,hidden_channel,1,1,bias=False)
        self.conv3 = nn.Conv2d(hidden_channel,hidden_channel,1,1,bias=False)
        self.conv4 = Conv(2*hidden_channel,out_channel,1,1)
        self.bn = nn.BatchNorm2d(2*hidden_channel)
        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.repeat_blocks = nn.Sequential(*[
            Bottleneck(hidden_channel,hidden_channel,shortcut,groups,expansion=1.0) for _ in range(repeats)
        ])

    def forward(self,x):
        y1 = self.conv3(self.repeat_blocks(self.conv1(x)))
        y2 = self.conv2(x)
        y3 = torch.cat((y1,y2),dim=1)
        return self.conv4(self.act(self.bn(y3)))
class SPP(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size_list = (5,9,11)):
        super().__init__()
        
        hidden_channel = in_channel //2
        self.conv1 = Conv(in_channel,hidden_channel,1,1)
        self.conv2 = Conv(hidden_channel*(len(kernel_size_list) +1), out_channel,1,1)
        self.spatial_pyramid_poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=kernel_size ,stride =1,padding=kernel_size//2) for kernel_size in kernel_size_list
        ])

    def forward(self,x):
        x = self.conv1(x)
        spp = torch.cat([x] +[m(x) for m in self.spatial_pyramid_poolings],dim=1)
        return self.conv2(spp)

class Focus(nn.Module):
    def __init__(self,in_channle,out_channel,kernel_size=1,stride=1,padding=None,groups=1,activation=True):
        super().__init__()
        self.conv = Conv(in_channle*4,out_channel,kernel_size,stride,padding,groups,activation)

    def forward(self,x):
        a = x[...,::2,::2]
        b = x[...,1::2,::2]
        c = x[...,::2,1::2]
        d = x[...,1::2,1::2]
        return self.conv(torch.cat([a,b,c,d],dim=1))

class Concat(nn.Module):
    def __init__(self,dimension=1):
        super().__init__()
        self.dimension = dimension
    def forward(self,x):
        return torch.cat(x,dim=self.dimension)

class Detect(nn.Module):
    def __init__(self,num_classes,num_anchor,reference_channles):
        super().__init__()
        self.num_anchor = num_anchor
        self.num_classes = num_classes
        self.num_output = self.num_classes +5
        self.heads = nn.ModuleList([
            nn.Conv2d(input_channel,self.num_output*self.num_anchor,1) for input_channel in reference_channles
        ])

    def forward(self,x):
        for ilevel,head in enumerate(self.heads):
            x[ilevel] = head(x[ilevel])
        return x

class Yolo(nn.Module):
    def __init__(self,num_classes,config_file):
        super().__init__()
        self.num_classes = num_classes
        self.model,self.saved_index,self.anchors = self.build_mode(config_file,num_classes)
    
    def forward(self,x):
        y = []
        for module_instance in self.model:
            if module_instance.from_index != -1:
                if isinstance(module_instance.from_index ,int):
                    x = y[module_instance.from_index]
                else:
                    xout = []
                    for i in module_instance.from_index:
                        if i == -1:
                            xvalue = x
                        else:
                            xvalue = y[i]
                        xout.append(xvalue)
                    x = xout
            x = module_instance(x)

            if module_instance.layer_index in self.saved_index:
                y.append(x)

            else:
                y.append(None)
        return x


    def parse_string(self,value):
        if value =="None":
            return None
        elif value == "True":
            return True
        elif value == "False":
            return False
        else:
            return value


    def build_mode(self,config_file,num_classes,input_channel=3):
        with open(config_file) as f:
            self.yaml = yaml.load(f,Loader=yaml.FullLoader)
        layers_cfg_list = self.yaml['backbone'] +self.yaml['head']
        anchors,depth_multiple,width_multiple = [self.yaml[item] for item in ['anchors','depth_multiple','width_multiple']]
        num_anchor_per_level = len(anchors[0])//2
        num_output_per_level = (5 +self.num_classes)*num_anchor_per_level
        layers_channel = [input_channel]

        layers = []
        saved_layer_index =[]
        for layer_index,(from_index, repeat, module_name, args) in enumerate(layers_cfg_list):
            args = [self.parse_string(item) for item in args]
            module_class_reference = eval(module_name)
            if repeat >1:
                repeat = max(round(repeat*depth_multiple),1)
            if module_class_reference in [Conv,Bottleneck,SPP,Focus,BottleneckCSP]:
                input_channel = layers_channel[from_index]
                output_channel = args[0]
                if output_channel != num_output_per_level:
                    output_channel = make_divisible(output_channel*width_multiple,8)#??
                
                args = [input_channel,output_channel,*args[1:]]
                if module_class_reference in [BottleneckCSP]:
                    args.insert(2,repeat)
                    repeat=1
            elif module_class_reference is Concat:
                output_channel = 0
                for index in from_index:
                    if index != -1:
                        index +=1
                    output_channel += layers_channel[index]#计算输出通道
            elif module_class_reference is Detect:
                reference_channel = [layers_channel[index+1] for index in from_index]#又因初始化有个三
                args = [num_classes,num_anchor_per_level,reference_channel]
            else:
                output_channel = layers_channel[from_index]#需要关注一下layer的变化

            
            if repeat >1:
                module_instance = nn.ModuleList([
                    module_class_reference(*args) for _ in range(repeat)
                ])
            else:
                module_instance = module_class_reference(*args)
            
            module_instance.from_index = from_index#是因为一个循环要用吗？
            module_instance.layer_index = layer_index
            layers.append(module_instance)#更新每一层
            layers_channel.append(output_channel)#更新layer_channel

            if not isinstance(from_index,list):
                from_index = [from_index]
            
            saved_layer_index.extend(filter(lambda x:x!=-1,from_index))
            # print(layer_index,'-',input_channel,'-',output_channel,'-',from_index)


        return nn.Sequential(*layers) ,  sorted(saved_layer_index),anchors  




if __name__ == "__main__":
    model = Yolo(20,"/home/shenlan02/shu/1108/yolov5-2.0/models/yolov5m.yaml")
    input = torch.zeros((1,3,640,640))
    y = model(input)

    print('...')
    print(y[0][0,0,0,0].item())
    print(y[0][0][0])
    # print(y[0][0][0].item())

    # print(torch.tensor(y).shape)
    
 

