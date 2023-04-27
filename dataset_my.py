import os 
import sys
import cv2
import random
import nn_utils
import sys_utils
from sys_utils import _single_instance_logger as logger

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as T


class VOCDataset:
    def __init__(self,augment,image_size,root):
        self.augment = augment
        self.image_size = image_size
        self.root = root
        self.border_fill_value = 114
        self.border_fill_tuple = 114,114,114
        self.label_map = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.num_classes = len(self.label_map)
        self.all_labeled_information = []
        cache_name = sys_utils.get_md5(root)
        self.bulid_and_cache(f"runs/dataset_cache/{cache_name}.cache")



    def bulid_and_cache(self,cache_file):
        if os.path.exists(cache_file):
            logger.info(f"load labels from cache:{cache_file}")
            self.load_labled_information_from_cache(cache_file)
        else:
            logger.info(f"build labels and save to cache:{cache_file}")

            self.bulid_labeled_inforamtion_and_save(cache_file)
    def load_labled_information_from_cache(self,cache_file):
        self.all_labeled_information = torch.load(cache_file)


    def bulid_labeled_inforamtion_and_save(self,cache_file):
        '''
        数据检测和缓存
        缓存：有效的图片路径，box信息，图像大小
        '''
        annotations_files = os.listdir(os.path.join(self.root,"Annotations"))
        annotations_files = list(filter(lambda x:x.endswith(".xml"),annotations_files))
        jpeg_files = [item[:-3]+'jpg' for item in annotations_files]

        annotations_files = map(lambda x: os.path.join(self.root,"Annotations",x),annotations_files)
        jpeg_files = map(lambda x: os.path.join(self.root,"JPEGImages",x),jpeg_files)

        for jpeg_file,annotations_files in zip(jpeg_files,annotations_files):
            pil_image = Image.open(jpeg_file)
            pil_image.verify()

            image_width,image_height = sys_utils.exif_size(pil_image)

            assert image_width >9 and image_height >9, f"image is too small{image_width}*{image_height}"
            pixel_annotations = nn_utils.load_voc_annotation(annotations_files,self.label_map)
            normalize_annotations = nn_utils.convert_to_normalize_annotation(pixel_annotations,image_width,image_height)
            self.all_labeled_information.append([jpeg_file,normalize_annotations,[image_width,image_height]])
        sys_utils.mkparents(cache_file)##??
        torch.save(self.all_labeled_information,cache_file)


    



    def __len__(self):
        return len(self.all_labeled_information)

    def __getitem__(self,image_indice):
        if self.augment:
            image,normalize_annotations = self.load_mosaic(image_indice)
            image = self.hsv_augment(image)
            if random.random()<0.5:
                image,normalize_annotations = self.horizontal_flip(image,normalize_annotations)
            return image,normalize_annotations
        else:
            return self.load_center_affine(image_indice)

            
    def load_center_affine(self,image_indice):
        image,normalize_annotations ,(width,height)= self.load_image_with_uniform_scale(image_indice)
        pad_width = self.image_size - width
        pad_height = self.image_size - height

        pad_left = pad_width //2
        pad_right = pad_width - pad_left
        pad_top = pad_height //2
        pad_bottom = pad_height - pad_top

        cv2.copyMakeBorder(image,pad_top,pad_bottom,pad_left,pad_right,borderType=cv2.BORDER_CONSTANT,value=self.border_fill_tuple)

        x_alpha = width /self.image_size
        x_beta = pad_left /self.image_size
        y_alpha = height /self.image_size
        y_beta = pad_top / self.image_size

        normalize_annotations[:,[0,1]] = normalize_annotations[:,[0,1]]*[x_alpha,y_alpha]+[x_beta,y_beta]
        normalize_annotations[:,[3,2]] = normalize_annotations[:,[2,3]]*[x_alpha,y_alpha]
        return image,normalize_annotations
  
    
    def horizontal_flip(self,image,normalize_annotations):
        image = cv2.flip(image,flipCode=1)
        normalize_annotations = normalize_annotations.copy()
        image_width = image.shape[1]
        # print(image_width)
        normalize_annotations[:,0]=(image_width-1)/image_width-normalize_annotations[:,0]
        return image,normalize_annotations

    
    def hsv_augment(self,image,hue_gain=0.15,saturation_gain=0.7,value_gain=0.4):
        hue_gain = np.random.uniform(-1,+1)*hue_gain + 1
        saturation_gain = np.random.uniform(-1,+1)*saturation_gain + 1
        value_gain = np.random.uniform(-1,+1)*value_gain + 1

        hue,saturation,value = cv2.split(cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
        dtype = image.dtype
        lut_base = np.arange(0,256)
        lut_hue = ((lut_base*hue_gain)%180).astype(dtype)
        lut_saturation = np.clip(lut_base*saturation_gain,0,255).astype(dtype)
        lut_value = np.clip(lut_base*value_gain,0,255).astype(dtype)

        changed_hue = cv2.LUT(hue,lut_hue)
        changed_saturation = cv2.LUT(saturation,lut_saturation)
        changed_value = cv2.LUT(value,lut_value)

        image_hsv = cv2.merge((changed_hue,changed_saturation,changed_value))
        return cv2.cvtColor(image_hsv,cv2.COLOR_HSV2BGR)



  
    def load_mosaic(self,image_indice):
        x_center = int(random.uniform(self.image_size*0.5,self.image_size*1.5))
        y_center = int(random.uniform(self.image_size*0.5,self.image_size*1.5))
        num_images = len(self.all_labeled_information)
        all_image_indices = [image_indice]+[random.randint(0,num_images-1) for _ in range(3)]#-1??
        alignment_corner_point = [
            [1,1],
            [0,1],
            [1,0],
            [0,0]
        ]
        merge_mosaic_image_size = self.image_size*2
        merge_mosaic_image = np.full((merge_mosaic_image_size,merge_mosaic_image_size,3),self.border_fill_value,dtype=np.uint8)
        merge_mosaic_pixel_annotations = []
        for  index , (image_indice,(corner_point_x,corner_point_y)) in enumerate(zip(all_image_indices,alignment_corner_point)):
            image,noarmalize_annotations ,(image_width,image_height) = self.load_image_with_uniform_scale(image_indice)
            #pingjieqian ,test
            nn_utils.draw_norm_bboxes(image,noarmalize_annotations,color=(0,0,255),thickness=5)
            # if index == 0:
            # noarmalize_annotations = np.zeros((0,5))
            corner_point_x = corner_point_x*image_width
            corner_point_y = corner_point_y*image_height

            x_offset = x_center - corner_point_x
            y_offset = y_center - corner_point_y

            M = np.array([
                [1,0,x_offset],
                [0,1,y_offset]
            ],dtype=np.float32)#这个dtype,不能丢
            cv2.warpAffine(image,M,(merge_mosaic_image_size,merge_mosaic_image_size),
                            borderMode=cv2.BORDER_TRANSPARENT,dst=merge_mosaic_image,
                            flags=cv2.INTER_NEAREST)
        # cv2.imwrite("merge_mosaic_image.jpg",merge_mosaic_image)
            pixel_annotations = nn_utils.convert_to_pixel_annotation(noarmalize_annotations,image_width,image_height)
            pixel_annotations = pixel_annotations +[x_offset,y_offset,x_offset,y_offset,0]#??
            merge_mosaic_pixel_annotations.append(pixel_annotations)
        # print(np.array(merge_mosaic_pixel_annotations).shape)
        # print(np.array(merge_mosaic_pixel_annotations))

        merge_mosaic_pixel_annotations = np.concatenate(merge_mosaic_pixel_annotations,axis=0)#??
        # print(merge_mosaic_pixel_annotations.shape)

        np.clip(merge_mosaic_pixel_annotations[:,:4],a_min=0,a_max=merge_mosaic_image_size-1,out=merge_mosaic_pixel_annotations[:,:4])
        
        scale =random.uniform(0.5,1.5)
        M = np.array([
            [scale,0,self.image_size*(0.5-scale)],
            [0,scale,self.image_size*(0.5-scale)]
            ],dtype=np.float32)
        merge_mosaic_image  = cv2.warpAffine(merge_mosaic_image,M,(self.image_size,self.image_size),
                            flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(self.border_fill_tuple))

        num_targets = len(merge_mosaic_pixel_annotations)

        output_normalize_annotations = np.zeros((0,5))

        if num_targets >0:
            targets_temp = np.ones((num_targets*2,3))
            targets_temp[:,:2] = merge_mosaic_pixel_annotations[:,:4].reshape(num_targets*2,2)
            (targets_temp@M.T).reshape(num_targets,4)
            merge_projection_pixel_annotations = merge_mosaic_pixel_annotations.copy()
            merge_projection_pixel_annotations[:,:4] = (targets_temp@M.T).reshape(num_targets,4)
            
            np.clip(merge_projection_pixel_annotations[:,:4],a_min=0,a_max=self.image_size-1,out=merge_projection_pixel_annotations[:,:4])

            projection_box_width = merge_projection_pixel_annotations[:,2] -merge_projection_pixel_annotations[:,0] +1
            projection_box_height = merge_projection_pixel_annotations[:,3] - merge_projection_pixel_annotations[:,1]+1
            orginal_box_width = merge_mosaic_pixel_annotations[:,2] - merge_mosaic_pixel_annotations[:,0]+1
            orginal_box_height = merge_mosaic_pixel_annotations[:,3] - merge_mosaic_pixel_annotations[:,1]+1
            area_projection = projection_box_width*projection_box_height
            area_original = orginal_box_width*orginal_box_height

            aspect_ratio = np.maximum(projection_box_width/(projection_box_height+1e-6),projection_box_height/(projection_box_width+1e-6))
            keep_indices = (projection_box_width >2)&\
                            (projection_box_height >2)&\
                            (area_projection/(area_original*scale +1e-6)>0.2)&\
                            (aspect_ratio<20)
            merge_projection_pixel_annotations = merge_projection_pixel_annotations[keep_indices]
            output_normalize_annotations = nn_utils.convert_to_normalize_annotation(merge_projection_pixel_annotations,self.image_size,self.image_size)
        return merge_mosaic_image,output_normalize_annotations


    def load_image_with_uniform_scale(self,image_indice):
        jpeg_file,normalize_annotations , (image_width,image_height)= self.all_labeled_information[image_indice]
        image = cv2.imread(jpeg_file)
        to_image_size_ratio = self.image_size/max(image.shape[:2])

        if to_image_size_ratio != 1:
            if not self.augment  and to_image_size_ratio <1:
                interp = cv2.INTER_AREA
            else:
                interp = cv2.INTER_LINEAR
            image = cv2.resize(image,(0,0),fx = to_image_size_ratio,fy=to_image_size_ratio,interpolation=interp)
        image_resized_height,image_resized_width = image.shape[:2]
        return image,normalize_annotations.copy(),(image_resized_width,image_resized_height)
    # def load_random(self):
    #     print(np.random.rand(1,5))

    @staticmethod
    def collate_fn(batch):
        images , normalize_annotations = zip(*batch)
        all_annotations = []
        all_images = []
        for index ,(annotations,image) in enumerate(zip(normalize_annotations,images)):
            num_targets = len(annotations)
            new_annotations = np.zeros((num_targets,6))
            new_annotations[:,0] = index
            new_annotations[:,1:] =annotations[:,[4,0,1,2,3]]
            all_annotations.append(new_annotations)
            all_images.append(T.to_tensor(image))
        visual_image_id = random.randint(0,len(images)-1)
        visual_image = images[visual_image_id]
        visual_annotations = normalize_annotations[visual_image_id]
        visual_info = visual_image_id,visual_image,visual_annotations

        all_annotations = np.concatenate(all_annotations,axis=0)
        all_annotations = torch.FloatTensor(all_annotations)
        all_images = torch.stack(all_images,dim=0)
        return all_images,all_annotations,visual_info




if __name__ == "__main__":
        root="/home/shenlan02/shu/1108/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/"
        dataset = VOCDataset(True,640,root)
        # image,normal,(w,h)=dataset.load_image_with_uniform_scale(0)
        # print(image.shape,w,h)
        # print(normal)
        # nn_utils.setup_seed(1)

        # image,ouput = dataset.load_mosaic(0)

        # pixel = np.array(
        #     [[100,50,200,150,0]]
        # )
        # nor = dataset.convert_to_normalize_annotation(pixel,640,640)
        # pix = dataset.convert_to_pixel_annotation(nor,640,640)
        # print(pix.tolist())

        # print(image.shape)
        # print(ouput)
        # nn_utils.draw_norm_bboxes(image,ouput,thickness=3)
        # cv2.imwrite("11.jpg",image)
        # print(np.random.rand(1,5))
        # cv2.imwrite("22.jpg",image)
        # dataset.load_random()


        # image,ouput = dataset.load_center_affine(0)
        # print(image.shape)
        # print(ouput)
        # nn_utils.draw_norm_bboxes(image,ouput,thickness=3)
        # cv2.imwrite("11.jpg",image)

        # image,out,= dataset.load_center_affine(12)
        # # dataset.hsv_augment(image)
        # # cv2.imwrite('33.jpg',image)
        # nn_utils.draw_norm_bboxes(image,out,color=(0,0.255),thickness=6)

        # image,out = dataset.horizontal_flip(image,out)
        # nn_utils.draw_norm_bboxes(image,out,thickness=3)
        # cv2.imwrite('44.jpg',image)

        # dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True,
            # pin_memory = True,num_workers=0,collate_fn=dataset.collate_fn)

        # for image,a,b in dataloader:
            # print(image.)

        # image,a,b = dataset.load_image_with_uniform_scale(0)
        # for images,a,b in dataloader:
        #     print(images.shape)
        #     print(a.shape)
        #     # print(b.shape)

        from tqdm import tqdm
        for _ in tqdm(dataset):
            pass
