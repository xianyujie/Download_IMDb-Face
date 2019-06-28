import os
import argparse
import cv2

def is_jpg(filename):
  data = open(filename,'rb').read(11)
  if data[:4] != b'\xff\xd8\xff\xe0' and data[:4]!=b'\xff\xd8\xff\xe1':
    return False
  if data[6:] != b'JFIF\0' and data[6:] != b'Exif\0':
    return False
  return True

def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaFace')
    # general
    parser.add_argument('--input', default='', help='',type=str)
    parser.add_argument('--err-output', default='', help='', type=str)
    parser.add_argument('--succ-output', default='', help='', type=str)
    parser.add_argument('--choose', default=0, help='if value is 0, just check. if value is 1, resave error jpg files', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    img_list_path = args.input
    with open(img_list_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(',') for line in lines]

    if args.choose == 0:
        err_jpg = open(args.err_output,'w')
        succ_jpg = open(args.succ_output,'w')
        cnt = 0
        for line in lines:
            cnt+=1
            print(len(lines), cnt)
            if not is_jpg(line[0]):
                err_jpg.write((',').join(line)+'\n')
            else:
                succ_jpg.write((',').join(line)+'\n')
        err_jpg.close()
        succ_jpg.close()
    elif args.choose==1:
        cnt = 0
        for line in lines:
            cnt += 1
            print(len(lines), cnt)
            if not is_jpg(line[0]):
                img = cv2.imread(line[0])
                cv2.imwrite(line[0],img)
    else:
        cnt = 0
        for line in lines:
            cnt += 1
            img = cv2.imread(line[0])

