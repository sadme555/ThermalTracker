# test_backbone.py
"""
骨干网络测试脚本 - 使用统一版本
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """运行骨干网络测试"""
    print("Testing unified backbone...")
    
    try:
        # 直接导入测试函数
        from models.backbone import test_backbone
        success = test_backbone()
        
        if success:
            print("\n🎉 Backbone is ready for use!")
            return True
        else:
            print("\n❌ Backbone tests failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)