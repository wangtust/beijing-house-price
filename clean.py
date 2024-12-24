import csv
import re

# 将楼层那列的汉字去除，只保留数字，删除含有nan，未知的行
# 输入文件路径
input_file = '/home/USER2022100821/work/Artificial-Intelligence/北京房价/new.csv'
# 输出文件路径
output_file = '/home/USER2022100821/work/Artificial-Intelligence/北京房价/new_cleaned.csv'

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='GB2312') as infile, open(output_file, 'w', encoding='GB2312', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # 读取表头
    header = next(reader)
    writer.writerow(header)
    
    # 逐行处理
    for row in reader:
        # 从右向左第11列是floor
        floor_index = -11
        # 去除floor列中的汉字及空格部分，只保留数字
        row[floor_index] = re.sub(r'[^\d]', '', row[floor_index])
        
        # 检查是否有缺失值、未知或nan
        if '' not in row and 'NA' not in row and 'nan' not in row and '未知' not in row:
            # 写入处理后的行
            writer.writerow(row)

print("Data cleaned and saved to new_cleaned.csv")