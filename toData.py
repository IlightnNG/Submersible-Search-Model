filename='testData.asc'

with open(filename, 'r') as f:  
    lines = f.readlines()  
  
# 去除每行首部的空格  
lines = [line.lstrip() for line in lines]  
  
# 将处理后的行写入新文件  
with open(filename, 'w') as f:  
    f.writelines(lines)
