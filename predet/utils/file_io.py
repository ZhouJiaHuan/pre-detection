# coding: utf-8
# Description: 文件及文件读取函数库
# Author: ZhouJH
# Date: 2019/10/24

import os
import shutil
import hashlib
from tqdm import tqdm


def get_file_path(file_dir, filter=[], sort=False):
    ''' 获取文件夹下文件路径列表

    遍历指定文件夹（包括子文件夹）下所有指定扩展名的文件路径，
    
    Args:
        file_dir: [str]，需要遍历的文件夹路径
        filter: [str list], 指定需要返回的扩展名列表，如'.txt', 若不指定，则返回所有文件路径
        sort: [bool], 是否对遍历结果进行排序（升序）

    Return:
        file_list: [str list]，包含所有指定扩展名的文件列表
    '''
    result=[]

    if not os.path.isdir(file_dir):
        print("no exist file directory: %s" % file_dir)

    for maindir, _, file_name_list in os.walk(file_dir):
        for filename in file_name_list:
            ext=os.path.splitext(filename)[-1]
            if filter == []:
                result.append(os.path.join(maindir,filename))
                continue
            elif ext in filter:
                result.append(os.path.join(maindir,filename))

    if sort is True:
        result.sort()
    return result


def parse_txt_to_list(txt_path, ignore_start='#'):
    '''将txt文本按行保存到列表中，并删除每行的空字符

    Args:
        txt_path: [str], 文件路径
        ignore_start: [str], 忽略以指定字符开头的行

    Return:
        字符串列表，每个字符串表示文本中的一行
    '''
    assert os.path.exists(txt_path), "{} not found!".format(txt_path)
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines() if not line.startswith(ignore_start)]


def parse_txt_to_dict(txt_path, split=':', ignore_start='#'):
    '''按行解析txt文本，并以key，value保存到字典中

    仅支持value为数字（默认将value转化为float）

    Args:
        txt_path: [str], txt文本路径
        split: [str], 切分key和value的字符
        ignore_start: [str], 忽略以指定字符开头的行
    Return:
        result: [dict], 解析出的键值对 
    '''
    assert os.path.exists(txt_path), "{} not found!".format(txt_path)
    result = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            if line.startswith(ignore_start):
                continue
            if split in line:
                key, value = line.strip().split(split)
                try:
                    result[key.strip()] = float(value.strip())
                except Exception as e:
                    print("parse failed for {}".format(line))
                    print(e)

    return result


def write_list_to_txt(str_list, txt_path):
    '''按行将list数据写入到txt中
    '''
    with open(txt_path, 'w') as f:
        f.write('\n'.join(list(map(str, str_list))))


def copy_with_list(path_list, dst_dir):
    '''根据路径列表拷贝文件

    Args:
        path_list: [str], 待拷贝的文件路径列表
        dst_dir: [str], 拷贝的目标文件夹
    '''
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for path in tqdm(path_list):
        try:
            shutil.copy(path, dst_dir)
        except Exception as e:
            print("copy failed for {}".format(path))
            print(e)


def remove_file(file_path, trash_dir='./trash_dir'):
    ''' 移除指定文件并备份到指定文件夹

    Args:
    file_path: [str], 待移除的文件路径
    trash_dir: [str], 移除文件的备份路径

    '''
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)
        #os.mkdir(trash_dir)
    if not os.path.exists(file_path):
        print("{0} not found!".format(file_path))
        return None

    file_name,file_ext=os.path.basename(file_path).split('.')

    # 备份路径下若已经存在同名文件，则自动修改命名（文件名_i.ext）
    i = 0
    while os.path.exists(os.path.join(trash_dir,file_name+'.'+file_ext)) and i<=3:
        #print("exist file in trash dir: ", file_name+'.'+file_ext)
        if i == 0:
            file_name = file_name + '_' + str(i+1)
        else:
            file_name = file_name[:-1] + str(i+1)
        i += 1
    file_trash_path = os.path.join(trash_dir,file_name+'.'+file_ext)
    shutil.move(file_path, file_trash_path)
    print("move file to {}".format(file_trash_path))


def get_md5(file_path, trunc=False):
    '''获取文件md5值

    Args:
        file_path: [str], 文件路径
        trunc: [bool], 是否只取前后部分数据进行计算（文件较大时速度更快）,只在文件大于100kb时起作用

    Return:
        [str], 文件md5值

    Exception:
        文件不存在时，抛出AssertionError
    '''
    assert os.path.exists(file_path), "{0} does not exist!".format(file_path)

    m = hashlib.md5()
    a_file = open(file_path, 'rb') # 使用二进制格式读取文件内容
    file_size = os.path.getsize(file_path)/1024
    if file_size > 100 and trunc==True:
        m.update(a_file.read(100))
        a_file.seek(-100,2)
        m.update(a_file.read())
    else:
        m.update(a_file.read())
        a_file.close()
    return m.hexdigest()


def make_dirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def compare_file_dir(dir1, dir2, result_txt, ext=['.xml', '.jpg']):
    '''

    查找dir2中与dir1下指定扩展名的同名文件，
    并将dir2中重复的文件名写到result_txt

    不支持文件夹递归比较！

    Args:
        dir1, dir2: 待比较的两个文件夹路径
        result_txt: dir2中重复文件名
        ext: 指定的扩展名列表

    Return:
        same_files: dir2中重复的文件名列表
    '''
    file_list1 = list(map(os.path.basename, get_file_path(dir1, ext)))
    file_list2 = list(map(os.path.basename, get_file_path(dir2, ext)))

    same_files = sorted(list(set(file_list1) & set(file_list2)))
    write_list_to_txt(same_files, result_txt)

    return same_files