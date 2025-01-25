# 加密流量异常检测系统

## 项目结构
```
project/
├── src/                # 源代码目录
├── data/               # 数据目录
├── models/             # 模型保存目录
├── logs/              # 日志目录
├── tests/             # 测试目录
├── requirements.txt   # 依赖包列表
└── README.md         # 项目说明文档
```

## 快速开始

1. 环境准备
```bash
# 克隆项目
git clone [项目地址]
cd project

# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

2. 运行setup.py
```bash
# 这将创建必要的目录结构并安装所有依赖
python setup.py
```

3. 运行程序
```bash
# 直接运行已知流量分类场景
python src/scenario1_known_traffic.py
```

## 输出说明

1. 日志输出
- 程序运行日志保存在 `logs/run.log`
- 包含训练过程、评估结果等详细信息

2. 结果输出
- 测试结果保存在 `results/scenario1/test_results.json`
- 可视化结果保存在 `results/scenario1/` 目录

## 注意事项
1. 确保Python版本 >= 3.8
2. 运行前请确保data目录中包含所需的数据文件
3. 如遇到中文显示问题，请确保系统支持UTF-8编码

