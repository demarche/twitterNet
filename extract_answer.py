import sys, os, os.path
import numpy as np
import glob
import MeCab
from progressbar import ProgressBar

print('loaded')

ROOT = "moved"
lines = open(ROOT+'\\user_info.txt', 'r').readlines()
user_max = len(lines)

for i1, line in enumerate(lines):
    info = line.split(",")
    user_id = info[0]
    RT_dev = float(info[9])

    corpus_path = ROOT+'\\corpus\\'+user_id+'.txt'

    #makedir
    root_, ext = os.path.splitext(corpus_path)
    dirpath = root_.replace(ROOT, "answer")
    os.makedirs(dirpath)

    #read cp
    cp_lines = open(corpus_path, 'r', encoding="utf-8", errors="ignore").readlines()

    print(user_id+":"+str(float(i1)/float(user_max)*100.0)+"%")
    p = ProgressBar(max_value = len(cp_lines)-1, min_value = 0)

    for i2, cp in enumerate(cp_lines):
        p.update(i2)
        cp_elem = cp.split(",")

        RT = float(cp_elem[2])

        ans = "0"
        if RT != 0 and RT < RT_dev:
            ans = "1"
        elif RT >= RT_dev and RT < RT_dev*2:
            ans = "2"
        elif RT >= RT_dev*2:
            ans = "3"

        with open(dirpath+"\\"+cp_elem[1]+".txt", 'w') as f:
            f.write(ans)
    p.finish()