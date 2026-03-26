#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI+中医知识图谱系统启动脚本
整合多模态处理、RAG查询、图谱可视化等功能
"""
import os
os.environ["MODELSCOPE_DISABLE_MODEL_WRAPPER"] = "1"

import os
import sys
import argparse
import subprocess
from pathlib import Path
from core.optimized_rag_engine import OptimizedRAGEngine
# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.config import Config
from core.graph_database import GraphDatabase


def check_dependencies():
    """检查依赖项"""
    print("=== 检查依赖项 ===")
    
    # 检查Neo4j连接
    try:
        config = Config()
        graph_db = GraphDatabase(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            database=config.NEO4J_DATABASE
        )
        print("✓ Neo4j数据库连接成功")
    except Exception as e:
        print(f"✗ Neo4j数据库连接失败: {e}")
        return False
    
    # 检查API密钥
    if not config.QWEN_API_KEY or config.QWEN_API_KEY == 'sk-0e62a83b088b44f1b1718c48395afbcf':
        print("✗ 请设置有效的千问API密钥")
        return False
    else:
        print("✓ 千问API密钥已配置")
    
    # 检查必要目录
    required_dirs = ['uploads', 'build', 'web/templates', 'web/static']
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 目录已创建: {dir_path}")
    
    return True


def build_knowledge_graph():
    """构建知识图谱"""
    print("=== 构建知识图谱 ===")
    
    try:
        # 运行知识图谱构建脚本
        build_script = project_root / "build" / "build_knowledge_graph.py"
        if build_script.exists():
            result = subprocess.run([sys.executable, str(build_script), "crawl"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ 知识图谱构建成功")
                return True
            else:
                print(f"✗ 知识图谱构建失败: {result.stderr}")
                return False
        else:
            print("✗ 找不到构建脚本")
            return False
    except Exception as e:
        print(f"✗ 构建知识图谱时出错: {e}")
        return False


def start_web_server():
    """启动Web服务器"""
    print("=== 启动Web服务器 ===")
    
    try:
        from web.app import app
        config = Config()
        
        # 强制关闭debug模式和监视进程
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
        
        print(f"服务器将在 http://{config.HOST}:{config.PORT} 启动")
        print("按 Ctrl+C 停止服务器")
        print("Debug模式: 已关闭")
        print("监视进程: 已关闭")
        
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=False,  # 强制关闭debug模式
            use_reloader=False  # 关闭文件监视器
        )
    except Exception as e:
        print(f"✗ 启动Web服务器失败: {e}")
        return False


def test_system():
    """测试系统功能"""
    print("=== 测试系统功能 ===")
    
    try:
        config = Config()
        graph_db = GraphDatabase(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            database=config.NEO4J_DATABASE
        )
        rag_engine = OptimizedRAGEngine(graph_db, config.QWEN_API_KEY)
        
        # 测试基本查询
        test_query = "头痛"
        print(f"测试查询: {test_query}")
        
        result = rag_engine.chat(test_query)
        answer = result.get("answer", "无答案")
        print(f"测试回答: {answer[:100] if isinstance(answer, str) else str(answer)[:100]}...")
        
        print("✓ 系统功能测试通过")
        return True
    except Exception as e:
        print(f"✗ 系统功能测试失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI+中医知识图谱系统")
    parser.add_argument("--check", action="store_true", help="检查依赖项")
    parser.add_argument("--build", action="store_true", help="构建知识图谱")
    parser.add_argument("--test", action="store_true", help="测试系统功能")
    parser.add_argument("--start", action="store_true", help="启动Web服务器")
    parser.add_argument("--all", action="store_true", help="执行完整流程")
    
    args = parser.parse_args()
    
    if not any([args.check, args.build, args.test, args.start, args.all]):
        # 默认执行完整流程
        args.all = True
    
    success = True
    
    if args.all or args.check:
        if not check_dependencies():
            success = False
            print("依赖项检查失败，请解决上述问题后重试")
            return
    
    if args.all or args.build:
        if not build_knowledge_graph():
            success = False
            print("知识图谱构建失败")
            return
    
    if args.all or args.test:
        if not test_system():
            success = False
            print("系统功能测试失败")
            return
    
    if args.all or args.start:
        if success:
            start_web_server()
        else:
            print("由于前面的步骤失败，跳过启动Web服务器")
    
    if args.all and success:
        print("\n=== 系统启动完成 ===")
        print("访问 http://localhost:5000 使用系统")


if __name__ == "__main__":
    main() 