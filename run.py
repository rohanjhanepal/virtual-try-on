import os
import time
from process import resize_img , pose_estimate , resize_cloth, mask_cloth,mask_img,parse_img , mask_img1 , mask_img_simple
import cv2
from matplotlib import pyplot as plt



def execute(person = "lady.jpg" , cloth= "cloth.jpg"):

    resize_img(person) 
    resize_cloth(cloth,person)
    parse_img(person)
    pose_estimate(person) 
    mask_cloth(person)
    mask_img1(person)
   
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle('Major Project : VIRTUAL CLOTH TRY ON USING NEURAL NETWORKS')
    rows = 2
    columns = 3
    
    if len(person)>0:
        with open('data//product_test.txt','w') as fp:
            final = person.split('.')[0]+'.jpg' + ' ' + person.split('.')[0] +'.jpg'
            fp.write(final)
            fp.write('\n')
    
    os.system('python test.py --name GMM --stage GMM  --checkpoint checkpoints/GMM/gmm_final.pth')
    time.sleep(2)
    os.system("python test.py --name TOM --stage TOM  --checkpoint checkpoints/TOM/tom_final.pth")

    img_final = cv2.imread("result//TOM//test//try-on//"+person.split('.')[0]+".jpg")
    img_cloth = cv2.imread("data//test//cloth//"+person.split('.')[0]+".jpg")
    img_person_mask = cv2.imread("data//test//image-mask//"+person.split('.')[0]+".png")
    img_person_seg = cv2.imread("data//test//image-parse//"+person.split('.')[0]+".png")
    img_before = cv2.imread("data//test//image//"+person.split('.')[0]+".jpg")
    img_pose = cv2.imread("result//TOM//test//im_pose//"+person.split('.')[0]+".jpg")
    
    imgs = [{'Person':img_before} , {'Cloth':img_cloth} , {'Person Mask':img_person_mask} , 
            {'Person Segmentation':img_person_seg} , {'Pose keypoints':img_pose}, {'final fitted image':img_final}]

    for i , item in enumerate(imgs):
        fig.add_subplot(rows, columns, int(i+1))
        plt.imshow(cv2.cvtColor(list(item.values())[0] , cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(list(item.keys())[0])

    
    cv2.imwrite('static/'+person.split('.')[0]+".jpg",img_final)
    cv2.imwrite('static/'+person.split('.')[0]+"_pose.jpg",img_pose)
    cv2.imwrite('static/'+person.split('.')[0]+"_segmentation.jpg",img_person_seg)
    cv2.imwrite('static/'+person.split('.')[0]+"_personMash.png",img_person_mask)

    cv2.imwrite('templates/'+person.split('.')[0]+".jpg",img_final)
    plt.savefig('static/my_plot.png')
    plt.savefig('templates/my_plot.png')
    

    
    # cv2.imshow('cloth' , img_cloth)
    # cv2.imshow('img_pose' , img_pose)
    
    # cv2.imshow('img_person_mask' , img_person_mask)
    # cv2.imshow('img_person_seg' , img_person_seg)
    
    # cv2.imshow('before' , img_before)

    # cv2.imshow('final' , img_final)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == '__main__':
    execute()