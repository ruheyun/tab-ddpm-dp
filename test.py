# coding:UTF-8
# RuHe  2026/3/13 10:35
import torch
import opacus

print(f"PyTorch Version: {torch.__version__}")
print(f"Opacus Version: {opacus.__version__}")

# 检查核心 API 是否可用
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# 测试 PrivacyEngine
try:
    engine = PrivacyEngine()
    print("PrivacyEngine available")
except Exception as e:
    print(f"PrivacyEngine error: {e}")

# 测试 make_private_with_epsilon 方法
try:
    assert hasattr(engine, 'make_private_with_epsilon')
    print("make_private_with_epsilon available")
except Exception as e:
    print(f"make_private_with_epsilon error: {e}")

# 测试 get_epsilon 方法
try:
    assert hasattr(engine, 'get_epsilon')
    print("get_epsilon available")
except Exception as e:
    print(f"get_epsilon error: {e}")