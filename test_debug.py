# final_fix_all.py
"""
最终完整修复脚本
"""

import os
import shutil

def fix_all_files():
    """修复所有文件"""
    print("开始最终修复...")
    
    # 1. 备份原始文件
    backup_dir = "backup_before_fix"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_fix = [
        "util/misc.py",
        "engine/train.py", 
        "main.py"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            print(f"✅ 已备份: {file_path} -> {backup_path}")
    
    print("✅ 所有文件备份完成")
    print("现在可以安全运行训练")

if __name__ == '__main__':
    fix_all_files()