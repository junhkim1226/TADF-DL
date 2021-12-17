import sys
import random

random.seed(0)

def make_txt(listname,filename):
    with open(filename,'w') as f:
        for line in listname:
            f.write(line)

f = open(sys.argv[1],'r')
g = open('train.txt','w')
h = open('val.txt','w')
k = open('test.txt','w')

lines = f.readlines()
random.shuffle(lines)
num_lines = len(lines)


train_list = lines[0:int(num_lines*0.8)]
val_list = lines[int(num_lines*0.8):int(num_lines*0.9)]
test_list = lines[int(num_lines*0.9):]

make_txt(lines, 'total.txt')
make_txt(train_list,'train.txt')
make_txt(val_list,'val.txt')
make_txt(test_list,'test.txt')
