from os import kill
from black import delimiter_split
import numpy as np
import torch
def test1(layers, large_scale, small_scale):
    def clip(coordinate):
        if coordinate < 0:
            return 0
        elif coordinate > 1:
            return 1
        else:
            return coordinate

    cell_num = np.array(layers)**2
    boxes = np.zeros([np.sum(cell_num),4,8],dtype=np.float32)
    sqrt2 = np.sqrt(2)
    length = 1.0/layers[0]
    for x in range(layers[0]):
        for y in range(layers[0]):
            for i in range(4):
                x_center = (x + 0.5) * length
                y_center = (y + 0.5) * length
                if i == 0:
                    box_width = box_height = small_scale[0]
                elif i == 1:
                    box_width = box_height = large_scale[0]
                elif i == 2:
                    box_width = large_scale[0] * sqrt2
                    box_height = large_scale[0] / sqrt2
                else:
                    box_width = large_scale[0] / sqrt2
                    box_height = large_scale[0] * sqrt2
                x_min = clip(x_center - box_width/2.0)
                x_max = clip(x_center + box_width/2.0)
                y_min = clip(y_center - box_height/2.0)
                y_max = clip(y_center + box_height/2.0)
                boxes[x*y,i,:] = [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]    
    return boxes

def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    def clip(coordinate):
        if coordinate < 0:
            return 0
        elif coordinate > 1:
            return 1
        else:
            return coordinate

    cell_num = np.array(layers)**2
    boxes = np.zeros([np.sum(cell_num),len(large_scale),8],dtype=np.float32)
    index_start = np.append([0],cell_num)
    sqrt2 = np.sqrt(2)
    for box in range(len(layers)):
        length = 1.0/layers[box]
        start = np.sum(index_start[:box+1])
        for x in range(1,layers[box]+1):
            for y in range(1,layers[box]+1):
                for i in range(4):
                    x_center = (x - 0.5) * length
                    y_center = (y - 0.5) * length
                    if i == 0:
                        box_width = box_height = small_scale[box]
                    elif i == 1:
                        box_width = box_height = large_scale[box]
                    elif i == 2:
                        box_width = large_scale[box] * sqrt2
                        box_height = large_scale[box] / sqrt2
                    else:
                        box_width = large_scale[box] / sqrt2
                        box_height = large_scale[box] * sqrt2
                    x_min = clip(x_center - box_width/2.0)
                    x_max = clip(x_center + box_width/2.0)
                    y_min = clip(y_center - box_height/2.0)
                    y_max = clip(y_center + box_height/2.0)
                    boxes[(x-1)*layers[box]+y-1+start,i,:] = [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]    
    return boxes#.reshape(np.sum(cell_num)*len(large_scale),8)


# a = default_box_generator([10,5,3,1],[0.2,0.4,0.6,0.8],[0.1,0.3,0.5,0.7])
# print(a[100,:,:])
# # import numpy as np
# import cv2
# # from torchvision import transforms
# # import matplotlib.pyplot as plt
# image = cv2.imread('CMPT733-Lab3-Workspace/data/train/images/03519.jpg')
# image = cv2.resize(image, (320,320))
# height, width, _ = image.shape
# boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
# # # print(boxs_default.shape)
# # #[x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
# # print(boxs_default[25])
# for j in range(134,135):
#     for i in range(4):
#         image_1 = image[:]
#         image_1 = cv2.rectangle(image_1, (int(boxs_default[j,i,4]*width),int(boxs_default[j,i,5]*height)), (int(boxs_default[j,i,6]*width),int(boxs_default[j,i,7]*height)), (255,0,0), 2)
# cv2.imwrite('test.png',image_1)

with open('CMPT733-Lab3-Workspace/data/train/annotations/00033.txt','r') as f:
    for line in f:
        class_id,x_s,y_s,w_box,h_box = line.split()
        class_id = int(class_id)
        x_s = float(x_s)
        y_s = float(y_s)
        w_box = float(w_box)
        h_box = float(h_box)
print(x_s)

#         image_1 = image[:]
#         image_1 = cv2.rectangle(image_1, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (255,0,0), 2)
#         cv2.imwrite('test.png',image_1)
        # cv2.waitKey(1000)
        # print(class_id,x_min,y_min,x_max,y_max)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# image = cv2.resize(image,(320,320))
# image_preprocess = transforms.Compose([transforms.ToTensor(),transforms.Resize([320,320])])
# image  = image_preprocess(image)
# image = cv2.resize(image,[320,320])
# # height, width, _ = image.shape
# print(image.shape)

# # image = np.transpose(image,[1,2,0])
# plt.imshow(image)
# plt.show()
import torch.nn as nn

class ConvBatchReLU(nn.Module):
    def __init__(self, cin, cout, k, s, pad=0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=cin,out_channels=cout,kernel_size=k,stride=s,padding=pad),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

# model = ConvBatchReLU(256,256,3,1)
# dummy = torch.randn(4, 256,3,3, requires_grad=True)
# out = model(dummy)
# print(out.shape)

# class ConvReshape(nn.Module):
#     def __init__(self,ksize=3,pad=1):
#         super().__init__()
#         self.conv = nn.Conv2d(256, 16, ksize, 1, pad)
    
#     def forward(self, x):
#         out_b = self.conv(x)
#         out_c = self.conv(x)
#         batch_size, channels, _, _ = out_b.shape
#         out_b = out_b.view([batch_size, channels,-1])
#         out_c = out_c.view([batch_size, channels,-1])
#         return out_b, out_c
# model2 = ConvReshape()
# d2 = torch.randn(4,256,5,5, requires_grad=True)
# out2,out3 = model2(d2)
# print(out2.shape)
# print(out3.shape)
import torch.nn.functional as F

def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    pred_confidence = pred_confidence.view(-1,4)
    pred_box = pred_box.view(-1,4)
    ann_confidence = ann_confidence.view(-1,4)
    ann_box = ann_box.view(-1,4)

    indices_obj = (ann_confidence[:,3]==0).nonzero().squeeze(1)
    indices_noobj = (ann_confidence[:,3]!=0).nonzero().squeeze(1)
    print('p',pred_confidence[indices_obj,:],'a',ann_confidence[indices_obj])
    loss_cls_obj = F.cross_entropy(pred_confidence[indices_obj,:],ann_confidence[indices_obj,:])
    loss_cls_noobj = F.cross_entropy(pred_confidence[indices_noobj,:],ann_confidence[indices_noobj,:])
    loss_cls = loss_cls_obj + 3*loss_cls_noobj

    loss_box = F.smooth_l1_loss(pred_box[indices_obj,:],ann_box[indices_obj,:])

    loss = loss_cls + loss_box
    return loss

# pred_confidence = torch.randn(4,5,4)
# pred_box = torch.randn(4,5,4)
# ann_confidence = torch.randn(4,5,4)
# ann_confidence[:,:,-1] = 1
# ann_confidence[:,2:5,1] = 1
# ann_confidence[:,2:5,-1] = 0
# ann_box = torch.randn(4,5,4)
# a = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
# print('a',a)

# with open("/Users/nemo/Documents/SFU/733/Assignments/A3/Assignment3-Py-Version/CMPT733-Lab3-Workspace/test.txt",'w') as f:
#     f.writelines([str(121)+' ','%.2f'%1.22222+' ', '%.2f'%2.34566+' ', '%.2f'%3.5644577+' ', '%.2f'%4.235325+'\n'])
#     f.writelines([str(121)+' ','%.2f'%1.22222+' ', '%.2f'%2.34566+' ', '%.2f'%3.5644577+' ', '%.2f'%4.235325+'\n'])
#     f.writelines([str(121)+' ','%.2f'%1.22222+' ', '%.2f'%2.34566+' ', '%.2f'%3.5644577+' ', '%.2f'%4.235325+'\n'])
# # f.close()
