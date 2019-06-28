import os,sys

def getURLList(csv_file):
    '''parse csv file, and return a list after simple-processing.'''
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(',') for line in lines]
    return lines[1:]

if __name__ == '__main__':
    csv_file = 'IMDb-Face.csv'
    img_root_dir = 'Img'
    image_lines = getURLList(csv_file)
    total_lines = len(image_lines)
    print('total_imdb:',total_lines)
    cnt = 0
    xx = []
    succ_f = open('success_file.txt','w')
    for i in range(total_lines):
        stdstr = '\r%s/%s'%(i,total_lines)
        sys.stdout.write(stdstr)
        sys.stdout.flush()

        name, index, image, rect, hw, url = image_lines[i]
        image_file = os.path.join(img_root_dir, '{}_{}'.format(name, index), image)
        x = '{}_{}_{}'.format(name,index,image)
        if os.path.exists(image_file) and x not in xx:
            xx.append(x)
            succ_f.write((',').join((image_file, rect, hw))+'\n')
            cnt+=1
    print('total images:',cnt)
    print('done!')
    succ_f.close()