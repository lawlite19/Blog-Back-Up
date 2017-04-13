#coding: utf-8
from PIL import Image
import os
import sys
import json
from datetime import datetime
from ImageProcess import Graphics

# 定义可以识别的图片文件类型，可以自行扩充
valid_file_type = ['jpg', 'png', 'gif']
# 定义压缩比，数值越大，压缩越小
SIZE_normal = 1.0
SIZE_small = 1.5
SIZE_more_small = 2.0


def make_directory(directory):
    """创建目录"""
    os.makedirs(directory)

def directory_exists(directory):
    """判断目录是否存在"""
    if os.path.exists(directory):
        return True
    else:
        return False

def list_img_file(directory):
    """列出目录下所有文件，并筛选出图片文件列表返回"""
    old_list = os.listdir(directory)
    # print old_list
    new_list = []
    for filename in old_list:
        name, fileformat = filename.split(".")
        if fileformat.lower() == "jpg" or fileformat.lower() == "png" or fileformat.lower() == "gif":
            new_list.append(filename)
    # print new_list
    return new_list


def print_help():
    print("""
    This program helps compress many image files
    you can choose which scale you want to compress your img(jpg/png/etc)
    1) normal compress(4M to 1M around)
    2) small compress(4M to 500K around)
    3) smaller compress(4M to 300K around)
    """)

def compress(choose, des_dir, src_dir, file_list):
    """压缩算法，img.thumbnail对图片进行压缩，还可以改变宽高数值进行压缩"""
    if choose == '1':
        scale = SIZE_normal
    if choose == '2':
        scale = SIZE_small
    if choose == '3':
        scale = SIZE_more_small
    for infile in file_list:
        img = Image.open(src_dir+infile)
        # size_of_file = os.path.getsize(infile)
        w, h = img.size
        img.thumbnail((int(w/scale), int(h/scale)))
        img.save(des_dir + infile)
def compress_photo():
    src_dir, des_dir = "photos/", "min_photos/"
    if directory_exists(src_dir):
        if not directory_exists(des_dir):
            make_directory(des_dir)
        # business logic
        file_list = list_img_file(src_dir)
        # print file_list
        if file_list:
            print_help()
            compress('3', des_dir, src_dir, file_list)   
        else:
            pass
    else:
        print("source directory not exist!")    

def handle_photo():
    src_dir, des_dir = "photos/", "min_photos/"
    file_list = list_img_file(src_dir)
    list_info = []
    if file_list:
        for i in range(len(file_list)):
            filename = file_list[i]
            date_str, info = filename.split("_")
            info, _ = info.split(".")
            date = datetime.strptime(date_str, "%Y-%m-%d")
            year_month = date_str[0:7]            
            if i == 0:  # 处理第一个文件
                new_dict = {"date": year_month, "arr":{'year': date.year,
                                                                       'month': date.month,
                                                                       'link': [filename],
                                                                       'text': [info],
                                                                       'type': ['image']
                                                                       }
                                            } 
                list_info.append(new_dict)
            elif year_month != list_info[-1]['date']:  # 不是最后的一个日期，就新建一个dict
                new_dict = {"date": year_month, "arr":{'year': date.year,
                                                       'month': date.month,
                                                       'link': [filename],
                                                       'text': [info],
                                                       'type': ['image']
                                                       }
                            }
                list_info.append(new_dict)
            else:  # 同一个日期
                list_info[-1]['arr']['link'].append(filename)
                list_info[-1]['arr']['text'].append(info)
                list_info[-1]['arr']['type'].append('image')
    list_info.reverse()  # 翻转
    final_dict = {"list": list_info}
    with open("../lawlite19.github.io/source/photos/data.json","w") as fp:
        json.dump(final_dict, fp)

def cut_photo():
    """裁剪算法，指定宽高数值进行裁剪"""
    src_dir, des_dir = "photos/", "min_photos/"
    width = 30
    height = 30 
    if directory_exists(src_dir):
        if not directory_exists(des_dir):
            make_directory(des_dir)
        # business logic
        file_list = list_img_file(src_dir)
        # print file_list
        if file_list:
            print_help()
            for infile in file_list:
                #img = Image.open(src_dir+infile)
                Graphics(infile=src_dir+infile, outfile=des_dir + infile).cut_by_ratio(width, height)            
        else:
            pass
    else:
        print("source directory not exist!")     



def git_operation():
    os.system('git add --all')
    os.system('git commit -m "add photos"')
    os.system('git push origin master')

if __name__ == "__main__":
    compress_photo()
    git_operation()
    handle_photo()
    #cut_photo()
    
    
    