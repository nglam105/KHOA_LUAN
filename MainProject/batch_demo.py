import argparse
import os
import numpy as np
import shutil
import glob
import cv2
import  subprocess

def DeleteAllFileinFolder(path):
    dir=os.listdir(path)
    for file in dir:
        print(file)
        os.remove(path+"/"+file)

def SalientObject(input):
    if "/" in input:
        image_name = input.split('/')[-1]
    DeleteAllFileinFolder('PoolNet/data/ECSSD/Imgs1')
    DeleteAllFileinFolder('PoolNet/results/run-1-sal-e1')
    shutil.copy(input,'PoolNet/data/ECSSD/Imgs1/'+image_name)
    file=open("E:/MainProject/PoolNet/data/ECSSD/test1.lst","w")
    file.write(image_name)
    file.close()
    #subprocess.Popen(r"C:/Users/nglam/anaconda3/python.exe",cwd='E:/MainProject/PoolNet')
    subprocess.call(["python","joint_main.py","--mode","test","--model","results/run-1/models/final.pth"
                            ,"--test_fold","results/run-1-sal-e1","--sal_mode","e"],cwd='PoolNet')

def ObjectSegmentation(input):
    DeleteAllFileinFolder('CenterMask/output')
    subprocess.call(["python","demo/centermask_demo.py","--config-file","configs/centermask/centermask_V_99_eSE_FPN_ms_3x.yaml","--weights","centermask-V2-99-FPN-ms-3x.pth"
                     ,"--conf_th","0.1","--display_text","True","--display_scores","True","--input",input,"--output_dir","output"],cwd='CenterMask')

def PhatHienVatTheKhongNoiBat(input,name_file):
    salientFolder= os.listdir('PoolNet/results/run-1-sal-e1')
    for img in salientFolder:
        salientMap=cv2.imread('PoolNet/results/run-1-sal-e1/'+img,0)
        ret,salientMap=cv2.threshold(salientMap,100,255,cv2.THRESH_BINARY)

    maskNotSalient=np.zeros(shape=(salientMap.shape[0],salientMap.shape[1],1),dtype=np.uint8)
    MaskObjectFolder=os.listdir('CenterMask/output')
    for img in MaskObjectFolder:
        if "mask_"in img:
            objectImg=cv2.imread('CenterMask/output/'+img,0)
            ret,objectImg=cv2.threshold(objectImg,100,255,cv2.THRESH_BINARY)
            Intersect = cv2.bitwise_and(objectImg, salientMap)
            num_intersect = np.count_nonzero(Intersect)
            num_objectImg = np.count_nonzero(objectImg)
            print(f"{img} {num_intersect} {num_objectImg}")
            if num_intersect/num_objectImg<0.7:
                maskNotSalient=cv2.bitwise_or(maskNotSalient,objectImg)

    pos=input.find('.')
    filetype=input[pos+1:]
    print("\n\n file dau vao"+name_file[:-4]+"\n\n")
    cv2.imwrite("generative_inpainting/result/"+name_file[:-4]+"_mask."+filetype,maskNotSalient)
    shutil.copy(input, "generative_inpainting/result/"+name_file[:-4]+"."+ filetype)

def Inpainting(input):
    pos = input.find('.')
    filetype = input[pos + 1:]
    path_input="result/input." + filetype
    path_mask="result/mask."+ filetype
    path_output="result/output."+ filetype
    subprocess.call(["python","test1.py","--image",path_input,"--mask",path_mask,
                     "--output",path_output,"--checkpoint_dir","model_logs/release_celebA_HQ_256"],cwd="generative_inpainting")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str,default='')
    args = parser.parse_args()
    DeleteAllFileinFolder('generative_inpainting/result')
    list_dir=os.listdir(args.input)
    for img in list_dir:
        if "_edge" not in img and "_result" not in img:
            print("\n\n\n"+img+"\n\n\n")
            input=args.input+"/"+img
            SalientObject(input)
            ObjectSegmentation(input)
            PhatHienVatTheKhongNoiBat(input,img)
            #Inpainting(args)
main()