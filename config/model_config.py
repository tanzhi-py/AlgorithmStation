#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型配置文件
管理本地模型路径和配置
"""

import os
from pathlib import Path

class ModelConfig:
    """模型配置类"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 模型目录
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # 支持的模型配置
    SUPPORTED_MODELS = {
        "text2vec-base-chinese": {
            "path": "text2vec-base-chinese",
            "dimension": 768,
            "language": "zh",
            "description": "中文文本向量模型，适合中文文本相似度计算"
        }
    }
    
    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """获取模型路径"""
        if model_name in cls.SUPPORTED_MODELS:
            model_config = cls.SUPPORTED_MODELS[model_name]
            if model_config["path"].startswith("sentence-transformers/") or model_config["path"].startswith("BAAI/"):
                # 在线模型
                return model_config["path"]
            else:
                # 本地模型
                return str(cls.MODELS_DIR / model_config["path"])
        else:
            # 自定义模型
            if os.path.exists(os.path.join(cls.MODELS_DIR, model_name)):
                return str(cls.MODELS_DIR / model_name)
            else:
                return model_name
    
    @classmethod
    def is_local_model(cls, model_name: str) -> bool:
        """判断是否为本地模型"""
        if model_name in cls.SUPPORTED_MODELS:
            model_config = cls.SUPPORTED_MODELS[model_name]
            return not (model_config["path"].startswith("sentence-transformers/") or model_config["path"].startswith("BAAI/"))
        else:
            # 自定义模型，检查本地是否存在
            return os.path.exists(os.path.join(cls.MODELS_DIR, model_name))
    
    @classmethod
    def get_model_dimension(cls, model_name: str) -> int:
        """获取模型向量维度"""
        if model_name in cls.SUPPORTED_MODELS:
            return cls.SUPPORTED_MODELS[model_name]["dimension"]
        else:
            # 默认维度
            return 768
    
    @classmethod
    def list_available_models(cls) -> list:
        """列出可用的本地模型"""
        available_models = []
        for model_name, config in cls.SUPPORTED_MODELS.items():
            if cls.is_local_model(model_name):
                model_path = cls.MODELS_DIR / config["path"]
                if model_path.exists():
                    available_models.append({
                        "name": model_name,
                        "path": str(model_path),
                        "dimension": config["dimension"],
                        "description": config["description"]
                    })
        return available_models
    
    @classmethod
    def ensure_models_dir(cls):
        """确保模型目录存在"""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        return cls.MODELS_DIR 