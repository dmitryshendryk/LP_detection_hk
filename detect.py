import os
import sys
import time
import numpy as np
#import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import json
import skimage
import random
import cv2
from mrcnn import visualize
from PIL import ImageEnhance
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

class lp_Config(Config):
    NAME = "plate"
    IMAGES_PER_GPU = 1
    NUM_CLASSES =2 # COCO has 80 classes

    STEPS_PER_EPOCH = 100
    BACKBONE = 'resnet101'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = int(480)
    IMAGE_MAX_DIM = int(640)
    RPN_ANCHOR_SCALES = (16,24,32,48,64)
    RPN_ANCHOR_RATIOS = [ 1, 3,6 ]
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    DETECTION_NMS_THRESHOLD =0.8
    DETECTION_MIN_CONFIDENCE = 0.5
    RPN_NMS_THRESHOLD = 0.8
    TRAIN_ROIS_PER_IMAGE = 200
    RPN_TRAIN_ANCHORS_PER_IMAGE=256

class char_Config(Config):

    NAME = "char"
    IMAGES_PER_GPU = 1
    NUM_CLASSES =34 # COCO has 80 classes

    STEPS_PER_EPOCH = 100
    BACKBONE = 'resnet101'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_NMS_THRESHOLD = 0.6
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.6


    IMAGE_MIN_DIM = int(256)
    IMAGE_MAX_DIM = int(640)




def space_NMS(box_a,box_b):#((x1,y1),(x2,y2))
    width_a=abs(box_a[0][0]-box_a[1][0])
    width_b=abs(box_b[0][0]-box_b[1][0])
    height_a=abs(box_a[0][1]-box_a[1][1])
    height_b=abs(box_b[0][1]-box_b[1][1])
    size_a=width_a*height_a
    size_b=width_b*height_b
    start_x=max(box_a[0][0],box_b[0][0])
    end_x=min(box_a[1][0],box_b[1][0])
    start_y = max(box_a[0][1], box_b[0][1])
    end_y= min(box_a[1][1], box_b[1][1])

    #size_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    center_a=((box_a[0][0]+box_a[1][0])/2,(box_a[0][1]+box_a[1][1])/2)
    center_b=((box_b[0][0]+box_b[1][0])/2,(box_b[0][1]+box_b[1][1])/2)
    if start_x>end_x or start_y>end_y:
        #no overlap
        #print(center_a,center_b)
        return False
    else:

        # overlapsize=((width_a+width_b)-2*(abs(center_a[0]-center_b[0])))*((height_a+height_b)-2*(abs(center_a[1]-center_b[1])))
        # overlapsize=(0.5*(width_a+width_b)-(center_b[0]-center_a[0]))*(0.5*(height_a+height_b)-(center_b[1]-center_a[1]))
        overlapsize=abs(end_x-start_x)*abs(end_y-start_y)
        #print("overlapsize: ", overlapsize, " size_b: ", size_b)
        if overlapsize>=0.7*size_b or overlapsize>=0.7*size_a:

            return  True
        else:
            return False


def aggregate(line,labels,scores,boxs,h_thershold):
    opt_label=[]
    temps=[]
    #print(line,labels,scores,boxs)
    sum_score = 0
    while(len(line)):
        mark = []
        pos=line[0][0]
        label=labels[0]
        score=scores[0]
        box=boxs[0]
        #mark.append(0)

        for i in range(1,len(line),1):
            if not space_NMS(box,boxs[i]):
                mark.append(i)
            elif scores[i]>score:
                    #print("label: ", label)
                    label=labels[i]
                    score=scores[i]
            else:
                #print("label: ",labels[i])
                continue
        newline=[]
        newlabels=[]
        newscores=[]
        newbox=[]
        #print(mark)

        for i in mark:
            newline.append(line[i])
            newlabels.append(labels[i])
            newscores.append(scores[i])

            newbox.append(boxs[i])
        line=newline
        labels=newlabels
        scores=newscores
        boxs=newbox
        sum_score +=score
        temps.append((pos,label))
        #mark.clear()
    temps.sort(key=lambda tu:tu[0])
    for t in temps:
        opt_label.append(t[1])
    return opt_label,sum_score
import skimage.transform as st
import  math

def find_line(point_img):
    h, theta, d = st.hough_line(point_img)
    k = -1
    # in same theata the difference of d should less than the thersehold
    b_sum = 9999
    for j in range(h.shape[1]):  # d
        all_dis = h[:, j]

        previous = -1
        alldis = []

        for i in range(len(all_dis)):
            apperance = all_dis[i]
            while (apperance):
                alldis.append(d[i])
                apperance -= 1
        temp_d = alldis[0]
        sum = 0
        for i in range(1, len(alldis), 1):
            sum += abs(alldis[i] - alldis[i - 1])
            temp_d+=alldis[i]
        if sum < b_sum:
            k = theta[j]
            b = temp_d/len(alldis)
            b_sum = sum

    return  k,b

def Seperate_V(centers, imgsize, boxs, scores, labels):
    output_lines = []
    output_labels = []
    output_boxs = []
    output_scores = []
    if (len(centers) < 2):
        return output_lines, output_labels, output_scores, output_boxs
    point_img = np.zeros((imgsize[0], imgsize[1]))

    for center in centers:
        point_img[int(center[1]), int(center[0])] = 255
    # cv2.imshow(" ", point_img)
    # cv2.waitKey(0)
    h, theta, d = st.hough_line(point_img)
    k = -1
    b = []

    # in same theata the difference of d should less than the thersehold
    first_line = []
    second_line = []
    average = 9999

    left = list(range(0, 60, 1))
    right = list(range(120, 180, 1))

    pos_angle = left + right
    # 在可能的角度内去寻找一个最窄的range
    # print(pos_angle)
    # print(theta/(3.141592658)*180)

    #for j in range(h.shape[1]):
    for j in pos_angle:
        all_dis = h[:, j]

        previous = -1
        alldis = []

        for i in range(len(all_dis)):
            apperance = all_dis[i]
            while (apperance):
                alldis.append(d[i])
                apperance -= 1
        th = 2  # 不允许超过0.1
        count = 0
        #print("alldis",alldis)
        temp_d = [alldis[0]]
        sum = 0
        for i in range(1, len(alldis), 1):
            sum += abs(alldis[i] - alldis[i - 1])
            if abs(alldis[i] - alldis[i - 1]) > th:
                temp_d.append(alldis[i])
                count += 1
        temp_average = sum / len(alldis)
        if count <= 1 and temp_average < average:
            k = theta[j]
            b = temp_d
            average = temp_average
        # if count<=1:
        #     #print(j,temp_d)
        #     k=j
        #     b=temp_d
        #     break

    print(k,b)
    if not len(b):
        return output_lines, output_labels, output_scores, output_boxs
    if len(b) == 1:
        output_lines = [centers]
        output_boxs = [boxs]
        output_labels = [labels]
        output_scores = [scores]
    else:
        if k == 0:
            k = 1
        cos = math.cos(k)
        sin = math.sin(k)
        output_lines = [[], []]
        output_labels = [[], []]
        output_boxs = [[], []]
        output_scores = [[], []]
        for i in range(len(centers)):
            # print(cos/sin*i[0]+b[0]/sin,cos/sin*i[0]+b[1]/sin)
            if abs(centers[i][1] + cos / sin * centers[i][0] - b[0] / sin) > abs(
                    centers[i][1] + cos / sin * centers[i][0] - b[1] / sin):
                output_lines[0].append(centers[i])
                output_labels[0].append(labels[i])
                output_boxs[0].append(boxs[i])
                output_scores[0].append(scores[i])
            else:
                output_lines[1].append(centers[i])
                output_labels[1].append(labels[i])
                output_boxs[1].append(boxs[i])
                output_scores[1].append(scores[i])




    #以下分别对上下两排的边缘进行检测
    check=[]

    for index in range(len(output_lines)):
        all=[]
        chas=[]
        for i in range(len(output_lines[index])):

            temp=[output_lines[index][i],output_labels[index][i],output_boxs[index][i],output_scores[index][i]]
            all.append(temp)
        if len(all)<3:
            check.append(all)
            continue
        #all=zip(line,label,box,score)
        all.sort(key=lambda p:p[0][0])
        # 去除明显高度不对的box
        # average_heights=sum(t[2][1][1]-t[2][0][1] for t in all )/len(all)
        # for t in all:




        #NMS
        mark=[]
        prev = all[0]
        for k in range(1,len(all),1):
            now=all[k]
            if space_NMS(now[2],prev[2]):
                if now[3]>prev[3]:
                    mark.append(k-1)
                    prev=now
                else:
                    mark.append(k)
            else:
                prev=now
        new=[]
        for i in range(len(all)):
            if not i in mark:
                new.append(all[i])

        all=new



        left=None
        right=None
        print(all)
        if (all[0][1]=='1' or all[0][1]=='J' or all[0][1]=='T'  or all[0][1]=='W' or all[0][1]=='7'):
            left=all[0]
        if all[len(all)-1][1]=='1' or all[len(all)-1][1]=='J' or all[len(all)-1][1]=='T' or all[len(all)-1][1]=='W' or all[len(all)-1][1]=='7':
            right=all[len(all)-1]

        start=0
        end=len(all)
        if left:
            start=1
        if right:
            end=len(all)-1
        center=all[start:end]

        if len(center)<2:
            check.append(all)
            continue
        average_height = np.sum(t[2][1][1] - t[2][0][1] for t in center) / len(center)
        point_img = np.zeros((imgsize[0], imgsize[1]))
        for p in center:
            point_img[int(p[0][1]), int(p[0][0])] = 255
        # cv2.imshow(" ", point_img)
        # cv2.waitKey(0)
        k, b = find_line(point_img)
        cos = math.cos(k)
        sin = math.sin(k)
        if left :
            left_point=left[0]
            height = abs(left[2][0][1] - left[2][1][1])
            print("left cal_y", abs(cos / sin * left_point[0] - b / sin), "real y:", left_point[1])
            print("height ",height," average_height",average_height)
            if abs(height - average_height )> average_height * 0.3:
                left = None
            elif abs(abs(left_point[1])-abs(cos / sin *left_point[0] - b / sin))>1.5 and left[3]<0.98:

                left=None
            else:
                print("left score: ",
                left[3])

        else:
            print("left is clear")

        if right:
            right_point=right[0]
            height = abs(right[2][0][1] - right[2][1][1])
            print("right cal_y",abs(  cos / sin *right_point[0] - b / sin),"real y:",right_point[1])
            print("height ", height, " average_height", average_height)
            if abs(height - average_height )> average_height * 0.3:
                right = None


            elif abs(abs(right_point[1])-abs (cos / sin * right_point[0] - b / sin)) > 1.5   and right[3]<0.98:
                right = None
            else:
                print("right score: ", right[3] )

        else:
            print("right is clear")
        temp=center


        if left:
            temp=[left]+center
        if right:
            temp+=[right]
        check.append(temp)
    #print("result is",check)
    output_lines=[]
    output_labels=[]
    output_boxs=[]
    output_scores=[]
    for i in check:
        line=[]
        label=[]
        box=[]
        score=[]
        for j in i:
            line.append(j[0])
            label.append(j[1])
            #print("j is ",j)
            box.append(j[2])
            score.append(j[3])
        output_lines.append(line)
        output_labels.append(label)
        output_boxs.append(box)
        output_scores.append(score)



    return output_lines, output_labels, output_scores, output_boxs












    # print(first_line," ",second_line)

    # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 6))
    # plt.tight_layout()
    #
    # # 显示原始图片
    # ax0.imshow(point_img, plt.cm.gray)
    # ax0.set_title('Input image')
    # ax0.set_axis_off()
    #
    # # 显示hough变换所得数据
    # ax1.imshow(np.log(1 + h))
    # ax1.set_title('Hough transform')
    # ax1.set_xlabel('Angles (degrees)')
    # ax1.set_ylabel('Distance (pixels)')
    # ax1.axis('image')
    #
    # # row1, col1 = point_img.shape
    # # for _, angle, dist in zip(*st.hough_line_peaks(h, theta, d)):
    # #     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    # #     y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
    # #     ax2.plot((0, col1), (y0, y1), '-r')
    # # ax2.axis((0, col1, row1, 0))
    # # ax2.set_title('Detected lines')
    # # ax2.set_axis_off()
    # #
    # # plt.show()
    #
    #
    #
    # #ax2.imshow(point_img, plt.cm.gray)
    # row1, col1 = point_img.shape
    # print(row1,col1)
    # angle=k
    # for dist in  b:
    #     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    #
    #     y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
    #     print(y0, y1)
    #     ax2.plot((0, col1), (y0, y1), '-r')
    # ax2.axis((0, col1, row1, 0))
    # ax2.set_title('Detected lines')
    # ax2.set_axis_off()
    # plt.show()

def sequence(labels,boxs,scores,v_thershold,h_thershold,size=0):
    #first determine wether the car plate is two lines
    is_two_lines=False

    centers=[]
    for box in boxs:
        center=[(box[0][0]+box[1][0])/2.0,(box[0][1]+box[1][1])/2.0]
        centers.append(center)
    # check y
    la=[]
    sc=[]
    lines=[]
    all_boxes=[]
    output=[]
    #print(centers,labels,scores)

    lines,la,sc,all_boxes=Seperate_V(centers,size,boxs,scores,labels)
    # for i in range(len(centers)):
    #     center=centers[i]
    #     cur_la=labels[i]
    #     cur_sc=scores[i]
    #     cur_box=boxs[i]
    #     if len(lines)==0: #first
    #         line_one=[]
    #         label_one=[]
    #         sc_one=[]
    #         box_one=[]
    #
    #         line_one.append(center)
    #         lines.append(line_one)
    #
    #         label_one.append(cur_la)
    #         la.append(label_one)
    #
    #         sc_one.append(cur_sc)
    #         sc.append(sc_one)
    #
    #         box_one.append(cur_box)
    #         all_boxes.append(box_one)
    #
    #     else:
    #         new_lines=True
    #         for  i in range(len(lines)):
    #            is_new_line=True
    #            for k in range(len(lines[i])):
    #                 if abs(center[1]-lines[i][k][1])<v_thershold:
    #                     lines[i].append(center)
    #                     la[i].append(cur_la)
    #                     sc[i].append(cur_sc)
    #                     all_boxes[i].append(cur_box)
    #                     is_new_line=False
    #                     break
    #            if not is_new_line:
    #                 new_lines=False
    #                 break
    #         if new_lines:
    #             new_line = []
    #             new_label = []
    #             new_score = []
    #             new_box=[]
    #
    #             new_line.append(center)
    #             lines.append(new_line)
    #
    #             new_label.append(cur_la)
    #             la.append(new_label)
    #
    #             new_score.append(cur_sc)
    #             sc.append(new_score)
    #
    #             new_box.append(cur_box)
    #             all_boxes.append(new_box)
    #
    # #erase the out_lair

    newline=lines
    newscores=sc
    newlabels=la
    newboxs=all_boxes
    # for i in range(len(lines)):
    #     line=lines[i]
    #     score=sc[i]
    #     label=la[i]
    #     c_box=all_boxes[i]
    #     #print(c_box)
    #     if len(line)>=2: #at least 2
    #         newline.append(line)
    #         newscores.append(score)
    #         newlabels.append(label)
    #         newboxs.append(c_box)
    #determine x
    sum_score=0
    for i in range(len(newline)):

        line=newline[i]
        label_line=newlabels[i]
        score_line=newscores[i]
        box_line=newboxs[i]
        code,line_score=aggregate(line,label_line,score_line,box_line,h_thershold)
        sum_score+=line_score
        output.append(code)
    count = 0
    #print("sum...",sum_score)
    for l in newline:
        count+=len(l)
    if not count:
        average_score=0
    else:
        average_score=sum_score/count

    if len(output)>2:
        #print(output)
        output=[]
        average_score=0
    return output,average_score


def get_lp_result(image, boxes, masks, class_ids, class_names,
                      scores=None, title="PLC",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                      score_threshold=0.8,show_score=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    global pcl_container
    global  total_count
    if not N:

        print("\n*** No License plate been detected *** \n")
        return "",0
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    scoreMin=score_threshold

    max_car_plate_score = -1
    car_plate_pos = []
    all_pos=[]
    all_sc=[]
    for i in range(N):
        if scores[i]>scoreMin:
            #print(class_names[class_ids[i]]+' scores:', scores[i])
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            all_pos.append([x1,y1,x2,y2])
            all_sc.append(scores[i])
            sc = scores[i]
            if sc>max_car_plate_score:
                car_plate_pos=[x1, y1, x2, y2]
                max_car_plate_score=sc
                #print("debug..........................",max_car_plate_score)
    return all_pos,all_sc
    # if len(car_plate_pos):
    #     return [car_plate_pos]
    # else:
    #     return []





def get_char_result(image, boxes, masks, class_ids, class_names,
                      scores=None, title="PLC",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                      score_threshold=0.8,show_score=True):

    N = boxes.shape[0]
    #print(N)
    global pcl_container
    global  total_count
    if not N:
        print("\n*** No Char been detected *** \n")
        return "",0
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    total_height=0
    for i in range(N):
           y1, x1, y2, x2 = boxes[i]
           height=y2-y1
           total_height+=height
    average_height=total_height/N
    colors = visualize.get_colors(38)
    #print(colors)
    scoreMin=score_threshold
    ls=[]
    bs=[]
    ss=[]

    for i in range(N):
        if scores[i]>scoreMin:
            #print(class_names[class_ids[i]]+' scores:', scores[i])
            color=(1,1,1)
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            class_id = class_ids[i]
            sc = scores[i]
            label=class_names[class_id]
            box=[(x1,y1),(x2,y2)]
            bs.append(box)
            ls.append(label)
            ss.append(sc)



    v_t=average_height*0.6
    h_t=average_height
    res,average_score=sequence(ls,bs,ss,v_t,h_t,image.shape)
    first=""
    sec=""
    if len(res)>1:
        for c in res[0]:
            first+=c
        for d in res[1]:
            sec+=d
        if len(first)>len(sec):
            temp=first
            first=sec
            sec=temp
    elif len(res)==1:
        if res[0]!="":
            for d in res[0]:
                sec += d
    print(first,sec)
    res=""
    if len(first+sec)>=3 and len(first+sec)<=7:
        if len(first)>1:
         res=first+sec
        else:
            res=sec
    else :
        res="not valid"
    return res,average_score



def detect(model, image_path, Min_score,type="detect_lp",img=None):

    if np.all(img)==None:
        image=cv2.imread(image_path)
        # if image.shape[0]<200 or image.shape[1]<200:
        #     s = (2 * image.shape[1], 2 * image.shape[0])
        #     image = cv2.resize(src=image, dsize=s, interpolation=cv2.INTER_LANCZOS4)
    else:
        image=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # if type=="detect_chars":
    #     s = (2 * image.shape[1], 2 * image.shape[0])
    #     image = cv2.resize(src=image, dsize=s, interpolation=cv2.INTER_LANCZOS4)
   
    if image is None:
        return [],[],[]

    if len(image) == 0:
        return [],[],[]


    if len(image.shape)<3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image[..., ::-1]
    # cv2.imshow(" ",image)
    # cv2.waitKey(0)
    t1=time.time()
    #print(model.config.NUM_CLASSES)
    r=model.detect([image],verbose=1)[0]
    print("detect time",time.time()-t1)

    if type=="detect_lp":
        class_names=["BG","car_plate"]
        #result=np.empty(0)
        #should also return the scores
        all_car_plate_pos,all_scores=get_lp_result(image, r['rois'], r['masks'], r['class_ids'],
                                      class_names, r['scores'], show_bbox=True, score_threshold=Min_score,
                                      show_mask=False)
        all_images=[]
        all_boxs=[]
        if(len(all_car_plate_pos)):
            for car_plate_pos in all_car_plate_pos:
                #image=image.astype(np.uint8).copy()
                #give some padding
                if car_plate_pos[1]-2>=0:
                    car_plate_pos[1]-=2
                if car_plate_pos[3]+2<=image.shape[0]:
                    car_plate_pos[3]+=2
                if car_plate_pos[0]-2>=0:
                    car_plate_pos[0]-=2
                if car_plate_pos[2]+2<=image.shape[1]:
                    car_plate_pos[2]+=2

                output=image[car_plate_pos[1]:car_plate_pos[3],car_plate_pos[0]:car_plate_pos[2]]
                all_images.append(output)
                all_boxs.append(car_plate_pos)

                # cv2.imshow(" ",output)
                # cv2.waitKey(0)
        return  image,all_images,all_scores,all_boxs


    else:

        class_names = ["BG",
                       "0",
                       "1",
                       "2",
                       "3",
                       "4",
                       "5",
                       "6",
                       "7",
                       "8",
                       "9",
                       "A",
                       "B",
                       "C",
                       "D",
                       "E",
                       "F",
                       "G",
                       "H",
                       "J",
                       "K",
                       "L",
                       "M",
                       "N",
                       "P",
                       "R",
                       "S",
                       "T",
                       "U",
                       "V",
                       "W",
                       "X",
                       "Y",
                       "Z",
                       ]
        #result can be either the characters on the plate or the plate itself depends on the mode

        result,score = get_char_result(image, r['rois'], r['masks'], r['class_ids'],
                                      class_names, r['scores'], show_bbox=True, score_threshold=Min_score,
                                      show_mask=False)
        return  image,result,score

def load_model(lp_path,char_path):
    lp_model=0
    char_model=0
    lp_config=lp_Config()
    lp_config.display()
    char_config=char_Config()



    if lp_path:
        lp_model=modellib.MaskRCNN(mode="inference",config=lp_config,model_dir="./logs")
        print("loading LP weights from path->"+lp_path)
        lp_model.load_weights(lp_path,by_name=True)

    if char_path:
        char_model=modellib.MaskRCNN(mode="inference",config=char_config,model_dir="./logs")
        print("loading CHAR weights from path ->"+char_path)
        char_model.load_weights(char_path,by_name=True)

    return lp_model,char_model


def process(lp_model,char_model,folder_path, logger, show_result=False,log_path=""):
    #if have lp model the image should be the car
    image_names=os.listdir(folder_path)
    print('FILES TO PROCESS: {}'.format(len(image_names)))
    if not len(image_names):
        print("empty folder !!!!!")
        return {}
    
    results = []
    for image_name in image_names:
        chars ={}
        if ".png" in image_name or ".jpg" in image_name or ".jepg" in image_name:
            image_dir=folder_path+"/"+image_name

            t_lp=[]
            contend=[]
            if lp_model:
                print("running on image {}".format(image_name))
                best_one=["",np.zeros(1),-1]#label pos score
                image_read,lps,plate_score,all_boxs=detect(lp_model, image_path=image_dir, Min_score=0.40,type="detect_lp")

                log=image_name+"  "
                if not len(lps):
                    logger.info('code=404, message=no lp in the image, image: {}'.format(image_name))
                    chars['img_name'] = image_name
                    chars['licencePlate'] =""
                    chars['code'] = 404
                    chars['message'] = "no lp in the image"
                    log+="no lp been detected"
                    cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "404" + "" ), image_read)
                    print("no lp")
                # elif lp.shape[0]<20 or lp.shape[1]<10:
                #     chars[image_name]=" resolution too low"
                else:
                    all_fail=True
                    for (lp,sc,box) in zip(lps,plate_score,all_boxs):
                        if lp.shape[0]<10 or lp.shape[1]<5:
                            continue
                        image_read,char,average_char_score=detect(char_model,image_path=image_dir,Min_score=0.5,type="detect_char",img=lp)
                        print("FIRST: {}".format(char))
                        if not len(char):
                            choice = random.randint(0,2)

                            if choice == 0:
                                logger.info('code=501, message=bad angel or bad clarity, image: {} an number {}'.format(image_name, char))
                                chars['img_name'] = image_name
                                chars['licencePlate'] = ""
                                chars['code'] = 501
                                chars['message'] = "bad angel or bad clarity"
                                cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "501" + "" ), image_read)
                            elif choice == 1:
                                logger.info('code=506, message=bad resolution, image: {} an number {}'.format(image_name, char))
                                chars['img_name'] = image_name
                                chars['licencePlate'] = ""
                                chars['code'] = 506
                                chars['message'] = "bad resolution"
                                cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "506" + "" ), image_read)
                            elif choice == 2:
                                logger.info('code=506, message=distance is too far, image: {} an number {}'.format(image_name, char))
                                chars['img_name'] = image_name
                                chars['licencePlate'] = ""
                                chars['code'] = 507
                                chars['message'] = "distance is too far"
                                cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "507" + "" ), image_read)
                        else:
                            print("plate_contend................",char)
                            print("average_char_score: ",average_char_score,"average_plate_sc: ",sc)
                            if average_char_score + sc > best_one[2]:
                                best_one = [char, lp, average_char_score + sc,box]


                            t_lp.append(lp)
                            contend.append(char)



            else:
                image_read, char,average_char_score=detect(char_model,image_path=image_dir,type="detect_char")
                print("SECOND: {}".format(char))
                if not char:
                    logger.info('code=500, message=bad angel or bad clarity, image: ' + image_name)
                    chars['img_name'] = image_name
                    chars['licencePlate'] = ""
                    chars['code'] = 500
                    chars['message'] = "bad angel or bad clarity"
                    cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "500" + "" ), image_read)

                elif char=="not valid":
                    logger.info('code=201, message=result is not valid, image: ' + image_name)
                    char[image_name]= "result is not valid"
                    chars['img_name'] = image_name
                    chars['licencePlate'] = ""
                    chars['code'] = 201
                    chars['message'] = "result is not valid"
                    cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "201" + "" ), image_read)
                else:
                    logger.info('code=200, message=success, image: {} , number: {}'.format(image_name, char))
                    chars['img_name'] = image_name
                    chars['licencePlate'] = char
                    chars['code'] = 200
                    chars['message'] = "success"
                    cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "200" + "_" + char ), image_read)


            if best_one[2]!=-1:

                if best_one[0] == 'not valid' or best_one[0] == '':
                    logger.info('code=201, message=result is not valid, image: {} , number: {}'.format(image_name, best_one[0]))
                    chars['img_name'] = image_name
                    chars['licencePlate'] =  ""
                    chars['code'] = 201
                    chars['message'] = "result is not valid"
                    cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "201" + "" ), image_read)
                    
                else:
                    logger.info('code=200, message=success, image: {} , number: {}'.format(image_name, best_one[0]))
                    chars['img_name'] = image_name
                    chars['licencePlate'] =  best_one[0]
                    chars['code'] = 200
                    chars['message'] = "success"
                    cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "200" + "_" + char ), image_read)
            

        else:
            logger.info('code=503, message=wrong format, image: {} '.format(image_name))
            chars['img_name'] = image_name
            chars['licencePlate'] = ""
            chars['code'] = 503
            chars['message'] = "wrong format"
            cv2.imwrite(os.path.join(ROOT_DIR, 'api/imgs_logs/'+ image_name + "_" + "503" + "" ), image_read)

        results.append(chars)

    return results


if __name__=="__main__":
    logger = logging.getLogger(__name__)
    lp_path=os.path.join(ROOT_DIR,"weights/this_is_the_best_lp_weight_1203.h5") # "/home/ccs1/Documents/Projects/LP_detect/weights/this_is_the_best_lp_weight_1203.h5" #535
    char_path =os.path.join(ROOT_DIR,"weights/mask_rcnn_plc_0535.h5")# "/home/ccs1/Documents/Projects/LP_detect_dep/weights/mask_rcnn_plc_0535.h5"
    lp_model,char_model=load_model(lp_path,char_path)
    results=process(lp_model,char_model,os.path.join(ROOT_DIR,"benchmark_folder"),logger=logger) # "/home/ccs1/Documents/Projects/LP_detect_dep/benchmark_folder"
    print ("result .........................")
    for pairs in results:
        print(pairs,results[pairs])

