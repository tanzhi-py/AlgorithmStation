"""
性能优化配置文件
用于配置RAG引擎的性能参数
"""

# 性能优化配置
PERFORMANCE_CONFIG = {
    # 快速模式配置
    'FAST_MODE_ENABLED': True,           # 启用快速模式
    'THINKING_MODE_ENABLED': False,      # 禁用思考模式
    'VISUALIZATION_ENABLED': False,      # 禁用可视化
    'DEBUG_OUTPUT_ENABLED': False,       # 禁用调试输出
    
    # 处理限制
    'MAX_PROCESSING_TIME': 15,           # 最大处理时间（秒）
    'MAX_QUERY_RESULTS': 5,              # 减少查询结果数量
    'VECTOR_SEARCH_LIMIT': 5,            # 减少向量检索数量
    
    # 缓存配置
    'ENABLE_CACHE': True,                # 启用缓存
    'CACHE_TIMEOUT': 1800,               # 缓存超时时间（30分钟）
    
    # 模型配置
    'MAX_TOKENS': 1024,                  # 减少最大token数
    'TEMPERATURE': 0.3,                  # 降低温度参数
}

def get_performance_config():
    """获取性能配置"""
    return PERFORMANCE_CONFIG.copy()

def is_fast_mode_enabled():
    """检查是否启用快速模式"""
    return PERFORMANCE_CONFIG.get('FAST_MODE_ENABLED', True)

def is_thinking_mode_enabled():
    """检查是否启用思考模式"""
    return PERFORMANCE_CONFIG.get('THINKING_MODE_ENABLED', False)

def is_visualization_enabled():
    """检查是否启用可视化"""
    return PERFORMANCE_CONFIG.get('VISUALIZATION_ENABLED', False) 