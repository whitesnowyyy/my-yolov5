import model_my
import dataset_my
import torch
import torch.nn as nn
import torch.optim as optim



class YoloHead(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.strides = [8,16,32]
        self.anchors = torch.tensor([
              [10,13, 16,30, 33,23],  # P3/8
            [30,61, 62,45, 59,119],  # P4/16
            [116,90, 156,198, 373,326]  # P5/32
            ]).view(3,3,2) / torch.FloatTensor(self.strides).view(3,1,1)
        self.num_anchor_per_level = self.anchors.size(1)
        self.anchor_t = 4.0
        self.offset_boundary = torch.FloatTensor([
            [1,0],
            [0,1],
            [-1,0],
            [0,-1]
        ])

        self.BCEClassification = nn.BCEWithLogitsLoss(reduction='mean')
        self.BCEObjectness = nn.BCEWithLogitsLoss(reduction='mean')
        self.balance = [4.0,1.0,0.4]

        self.box_weight = 0.05
        self.objectness_weight  =1.0
        self.classification_weight = 0.5*self.num_classes /80#coco的类别数


    def forward(self,predict,targets):

        num_target = targets.size(0)
        loss_box_regression = torch.FloatTensor([0])
        loss_classification = torch.FloatTensor([0])
        loss_objectness = torch.FloatTensor([0])


        for  ilayer,layer in enumerate(predict):
            layer_height,layer_width = layer.shape[-2:]
            layer  = layer.view(-1,self.num_anchor_per_level,5+self.num_classes,layer_height,layer_width)
            layer = layer.permute(0,1,3,4,2).contiguous()
            
            feature_size_gain = targets.new_tensor([1,1,layer_height,layer_width,layer_height,layer_width])
            targets_feature_scale = targets*feature_size_gain
            
            anchors = self.anchors[ilayer]
            num_anchor = anchors.size(0)

            anchors_wh = anchors.view(num_anchor,1,2)
            targets_wh = targets_feature_scale[:,[4,5]].view(1,num_target,2)

            wh_ratio = targets_wh / anchors_wh
            max_wh_ratio_values,_  = torch.max(wh_ratio,1/wh_ratio).max(dim=2)

            select_mask = max_wh_ratio_values < self.anchor_t
            select_targets = targets_feature_scale.repeat(num_anchor,1,1)[select_mask]#??

            matched_num_target = len(select_targets)

         

            if matched_num_target >0:
                # select_anchor = anchors.view(num_anchor,1,2).repeat(1,num_target,1)[select_mask]#??
                select_anchor_index = torch.arange(num_anchor).view(num_anchor,1).repeat(1,num_target)[select_mask]

            select_targets_xy = select_targets[:,[2,3]]
            xy_divided_one_remainder = select_targets_xy %1.0
            coord_cell_middle = 0.5
            feature_map_low_boundary = 1.0
            feature_map_high_boundary = feature_size_gain[[2,3]]-1.0

            less_x_matched,less_y_matched = ((xy_divided_one_remainder< coord_cell_middle)&(select_targets_xy >feature_map_low_boundary)).T#z转置？？
            greater_x_matched,greater_y_matched = ((xy_divided_one_remainder>(1- coord_cell_middle))&(select_targets_xy < feature_map_high_boundary)).T

            select_anchor_index = torch.cat([
                select_anchor_index,
                select_anchor_index[less_x_matched],
                select_anchor_index[less_y_matched],
                select_anchor_index[greater_x_matched],
                select_anchor_index[greater_y_matched]
            ],dim=0)#??形状一样？？

            select_targets = torch.cat([
                select_targets,
                select_targets[less_x_matched],
                select_targets[less_y_matched],
                select_targets[greater_x_matched],
                select_targets[greater_y_matched]
            ],dim=0)

            xy_offset = torch.zeros_like(select_targets_xy)
            xy_offset = torch.cat([
                xy_offset,
                xy_offset[less_x_matched]+self.offset_boundary[0],
                xy_offset[less_y_matched]+self.offset_boundary[1],
                xy_offset[greater_x_matched]+self.offset_boundary[2],
                xy_offset[greater_y_matched]+self.offset_boundary[3]
            ])*coord_cell_middle

            matched_extend_num_target = len(select_targets)
            gt_image_id, gt_class_id = select_targets[:,[0,1]].long().T
            gt_xy = select_targets[:,[2,3]]
            gt_wh = select_targets[:,[4,5]]
            grid_xy = (gt_xy - xy_offset).long()


            grid_x,grid_y = grid_xy.T

            gt_xy = gt_xy - grid_xy

            select_anchors = anchors[select_anchor_index]
            object_predict = layer[gt_class_id,select_anchor_index,grid_y,grid_x]#???
            object_predict_xy =object_predict[:,[0,1]].sigmoid()*2.0 - 0.5
            object_predict_wh = torch.pow(object_predict[:,[2:3]].sigmoid()*2.0,2.0)*select_anchors#???

            object_predict_box = torch.cat((object_predict_xy,object_predict_wh),dim=1)
            object_ground_truth_box = torch.cat((gt_xy,gt_wh),dim=1)

            gious = self.giou(object_predict_box,object_ground_truth_box)
            giou_loss = 1 - gious

            loss_box_regression  += giou_loss.mean()
            
            featuremap_objectness = layer[...,4]
            objectness_ground_truth = torch.zeros_like(featuremap_objectness)

            objectness_ground_truth[gt_image_id,select_anchor_index,grid_y,grid_x] = gious.detch().clamp(0)#??

            if self.num_classes >1:
                object_classification = object_predict[:,5:]
                classification_targets = torch.zeros_like(object_classification)
                classification_targets[torch.arange(matched_extend_num_target),gt_class_id] = 1.0#？
                loss_classification += self.BCEClassification(object_classification,classification_targets)#??

        loss_objectness  += self.BCEObjectness(featuremap_objectness,objectness_ground_truth)*self.balance[ilayer]
        num_level = len(predict)
        scale = 3 /num_level

        batch_size = predict[0].shape[0]
        loss_box_regression *= self.box_weight*scale
        loss_objectness *= self.objectness_weight *scale
        loss_classification *= self.classification_weight *scale

        loss = loss_box_regression + loss_objectness + loss_classification
        return loss*batch_size





    def giou(self,a,b):
        a_xmin,a_xmax = a[:,0]-(a[:,2]-1)/2,a[:,0]+(a[:,2]-1)/2
        a_ymin,a_ymax = a[:,1]-(a[:,3]-1)/2,a[:,1]+(a[:,3]-1)/2
        b_xmin,b_xmax = b[:,0]-(b[:,2]-1)/2,b[:,0]+(b[:,2]-1)/2
        b_ymin,b_ymax = b[:,1]-(b[:,3]-1)/2,b[:,1]+(b[:,3]-1)/2

        inter_xmin = torch.max(a_xmin,b_xmin)
        inter_xmax = torch.min(a_xmax,b_xmax)
        inter_ymin = torch.max(a_ymin,b_ymin)
        inter_ymax = torch.min(a_ymax,b_ymax)

        inter_width = (inter_xmax-inter_xmin +1).clamp(0)
        inter_height = (inter_ymax - inter_ymin +1).clamp(0)
        inter_area = inter_width * inter_height
        union = a[:,2]*a[:,3] +b[:,2]*[b:3] - inter_area
        iou = inter_area / union

        convex_width = torch.max(a_xmax, b_xmax) - torch.max(a_xmin,b_xmin) + 1
        convex_height = torch.max(a_ymax,b_ymax) - torch.max(a_ymin,b_ymin) +1
        convex_area = convex_width * convex_height

        return iou - (convex_area - union )/convex_area



def train():
    train_set = dataset_my.VOCDataset(True,640,"/home/shenlan02/shu/1108/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/")
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=3,
        num_workers=0,shuffle=True,pin_memory=True,
        collate_fn=train_set.collate_fn)
    
    device = 'cuda:0'
    head = YoloHead(20).to(device)
    model = model_my.Yolo(train_set.num_classes,"/home/shenlan02/shu/1108/yolov5-2.0/models/yolov5s.yaml").to(device)
    optimizer = optim.SGD(model.parameters(),1e-2,0.9)
    for batch_index ,(images,targets,visual) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        predict = model(images)
        loss = head(predict,targets)
        print(loss)
        pass

if __name__ =="__main__":
    train()


