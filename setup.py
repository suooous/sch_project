import os
import subprocess
from setuptools import setup, find_packages

def create_project_structure():
    """创建项目所需的目录结构"""
    directories = [
        'data/processed',
        'results',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def install_requirements():
    """安装requirements.txt中的依赖"""
    try:
        print("开始安装依赖...")
        # 读取requirements.txt并过滤掉注释和空行
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
        
        # 逐个安装依赖
        for req in requirements:
            print(f"正在安装: {req}")
            subprocess.check_call(['pip', 'install', req])
        
        print("依赖安装完成")
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {str(e)}")
        raise
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    # 1. 创建目录结构
    create_project_structure()
    
    # 2. 安装依赖
    install_requirements()
    
    # 3. 设置包信息
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    setup(
        name='encrypted_traffic_detection',
        version='1.0.0',
        description='基于改进孤立森林的加密流量异常检测系统',
        author='Guojie Wang',
        author_email='1934116641@qq.com',
        packages=find_packages(),
        install_requires=requirements,
        python_requires='>=3.8',
    ) 