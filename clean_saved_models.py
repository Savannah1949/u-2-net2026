import os
# 定义目录路径
dir_path = "E:\\U-2-Net-master\\saved_models\\U2NETP"

# 列出目录中的所有文件
files = os.listdir(dir_path)


# 统计要删除的文件数量
delete_count = 0

print(f"开始清理 {dir_path} 目录...")

print("-" * 50)

# 遍历文件并删除不需要的文件
for file in files:
    file_path = os.path.join(dir_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
        delete_count += 1
        print(f"已删除: {file}")

print("-" * 50)
print(f"清理完成！")
print(f"共删除 {delete_count} 个文件")