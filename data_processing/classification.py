from Gray_level_co_occurrence_matrix import *
import pandas as pd
import numpy as np

imgpath = './data/2.0/IMG20190614105701.jpg'

newpath = './img/new'

csvpath = './new/data.csv'

cut(imgpath, 10, newpath)
print(newpath+imgpath[10:])

result = [str(item) for item in test(newpath+imgpath[10:])]

reader = pd.read_csv(csvpath)

asm = np.array(reader['asm'])
#print(asm)
con = np.array(reader['con'])
eng = np.array(reader['eng'])
idm = np.array(reader['idm'])

label = np.array(reader['label'])


asm_min = 1.0
for i in range (4) :
     n = asm_min
     asm_min = min(asm_min, abs(float(result[0]) - float(asm[i])))
     if(n == asm_min):
          break
#print(asm_min,i)
label_asm = label[i]
print(label_asm)

con_min = 1
for i in range (4) :
     n = con_min
     con_min = min(con_min, abs(float(result[1]) - float(con[i])))
     if(n == con_min):
          break
#print(con_min,i)
label_con = label[i]
print(label_con)

eng_min = 1
for i in range (4) :
     n = eng_min
     eng_min = min(eng_min, abs(float(result[2]) - float(eng[i])))
     if(n == eng_min):
          break
#print(eng_min,i)
label_eng = label[i]
print(label_eng)

idm_min = 1
for i in range (4) :
     n = idm_min
     idm_min = min(idm_min, abs(float(result[3]) - float(idm[i])))
     if(n == idm_min):
          break
#print(idm_min,i)
label_idm = label[i]
print(label_idm)

# label_image=1.0
# if(label_asm == label_con):
#      label_image = label_asm
#      if(label_image != label_eng)

# sample = [con_min,label_con,label_eng,label_idm] #出错了
# print(max(set(sample), key=sample.count))

