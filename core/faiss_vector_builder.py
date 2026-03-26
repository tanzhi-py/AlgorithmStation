#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于FAISS的向量构建系统
使用FAISS进行高效的向量存储和检索
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import torch

# 导入模型配置
try:
    from config.model_config import ModelConfig
except ImportError:
    # 如果导入失败，创建一个简单的配置类
    class ModelConfig:
        @staticmethod
        def get_model_path(model_name: str) -> str:
            return f"models/{model_name}"
        @staticmethod
        def is_local_model(model_name: str) -> bool:
            return True
        @staticmethod
        def get_model_dimension(model_name: str) -> int:
            return 768


class FAISSVectorBuilder:
    """
    基于FAISS的向量构建器
    
    设计理念：
    - 向量检索的目的是精确匹配节点名称
    - 大模型已经完成了问题分类和类型识别
    - 因此直接使用节点名称进行向量化，不添加类型前缀
    - 检索时也不按类型过滤，返回所有相似节点
    - 这样可以提高检索精度，减少噪声，提高效率
    """
    
    def __init__(self, model_name: str = "text2vec-base-chinese", 
                 use_local_model: bool = None):
        """
        初始化FAISS向量构建器
        
        Args:
            model_name: 句子嵌入模型名称（默认使用中文模型）
            use_local_model: 是否使用本地模型文件（None表示自动判断）
        """
        self.model_name = model_name
        
        # 自动判断是否使用本地模型
        if use_local_model is None:
            self.use_local_model = ModelConfig.is_local_model(model_name)
        else:
            self.use_local_model = use_local_model
            
        self.model = None
        self.index = None
        self.node_mapping = {}  # 节点ID到索引的映射
        self.reverse_mapping = {}  # 索引到节点ID的映射
        
        # 从配置获取向量维度
        self.vector_dim = ModelConfig.get_model_dimension(model_name)
        
        self.index_file = "build/faiss_index.bin"
        self.mapping_file = "build/node_mapping.pkl"
        
        # 支持的节点类型
        self.node_types = [
            "方名",      # 方剂名称
            "功能主治",   # 功效描述
            "中药名",     # 中药名称
            "来源",       # 方剂来源
            "别名",       # 方剂别名
            "处方",       # 处方组成
            "剂量"        # 中药剂量
        ]
        
        # 确保构建目录存在
        os.makedirs("build", exist_ok=True)
        
        self._load_model()
    
    def _load_model(self):
        """加载句子嵌入模型"""
        try:
            print(f"正在加载向量模型: {self.model_name}")
            
            # 获取模型路径
            model_path = ModelConfig.get_model_path(self.model_name)
            
            if self.use_local_model:
                # 使用本地模型路径
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"本地模型文件 {model_path} 不存在")
                print(f"从本地路径加载模型: {model_path}")
                self.model = SentenceTransformer(model_path)
            else:
                print(f"从HuggingFace Hub下载模型: {model_path}")
                self.model = SentenceTransformer(model_path)
            
            # 获取实际的向量维度
            actual_dim = self.model.get_sentence_embedding_dimension()
            self.vector_dim = actual_dim
            print(f"向量模型加载完成，维度: {self.vector_dim}")
            
            # 验证维度是否匹配配置
            expected_dim = ModelConfig.get_model_dimension(self.model_name)
            if expected_dim != actual_dim:
                print(f"警告：模型实际维度({actual_dim})与配置维度({expected_dim})不匹配")
                
        except Exception as e:
            print(f"向量模型加载失败: {e}")
            raise
    
    def get_node_text(self, node_type: str, node_name: str, properties: Dict = None) -> str:
        """
        直接返回节点名称用于向量化
        
        Args:
            node_type: 节点类型（保留参数以兼容接口）
            node_name: 节点名称
            properties: 节点属性（保留参数以兼容接口）
            
        Returns:
            节点名称（直接用于向量化）
        """
        # 直接返回节点名称，不添加类型前缀
        # 因为向量检索只是为了完全对应节点名称
        return node_name
    
    def build_faiss_index(self, graph_db) -> bool:
        """
        构建FAISS索引
        
        Args:
            graph_db: 图数据库连接
            
        Returns:
            是否构建成功
        """
        print("=== 开始构建FAISS索引 ===")
        
        try:
            # 收集所有节点
            all_nodes = []
            node_texts = []
            node_ids = []
            
            for node_type in self.node_types:
                print(f"正在处理 {node_type} 节点...")
                
                # 查询该类型的所有节点
                query = f"""
                MATCH (n:{node_type})
                RETURN n.name as name, n as node
                """
                
                results = graph_db.run(query).data()
                
                if not results:
                    print(f"未找到 {node_type} 类型的节点")
                    continue
                
                for result in results:
                    node_name = result['name']
                    node = result['node']
                    node_id = f"{node_type}:{node_name}"
                    
                    # 生成用于向量化的文本
                    text = self.get_node_text(node_type, node_name, dict(node))
                    
                    all_nodes.append({
                        'id': node_id,
                        'type': node_type,
                        'name': node_name,
                        'text': text
                    })
                    node_texts.append(text)
                    node_ids.append(node_id)
            
            if not node_texts:
                print("没有找到任何节点")
                return False
            
            print(f"找到 {len(node_texts)} 个节点，开始生成向量...")
            
            # 批量生成向量
            embeddings = self.model.encode(node_texts, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy()
            
            # 创建FAISS索引
            print("正在创建FAISS索引...")
            self.index = faiss.IndexFlatIP(self.vector_dim)  # 内积索引，用于余弦相似度
            
            # 归一化向量（用于余弦相似度）
            faiss.normalize_L2(embeddings)
            
            # 添加向量到索引
            self.index.add(embeddings.astype('float32'))
            
            # 创建节点映射
            for i, node_id in enumerate(node_ids):
                self.node_mapping[node_id] = i
                self.reverse_mapping[i] = all_nodes[i]
            
            print(f"FAISS索引构建完成，共 {self.index.ntotal} 个向量")
            
            # 保存索引和映射
            self.save_index()
            
            return True
            
        except Exception as e:
            print(f"构建FAISS索引失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_index(self):
        """保存FAISS索引和节点映射"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, self.index_file)
            print(f"FAISS索引已保存到: {self.index_file}")
            
            # 保存节点映射
            with open(self.mapping_file, 'wb') as f:
                pickle.dump({
                    'node_mapping': self.node_mapping,
                    'reverse_mapping': self.reverse_mapping
                }, f)
            print(f"节点映射已保存到: {self.mapping_file}")
            
        except Exception as e:
            print(f"保存索引失败: {e}")
    
    def load_index(self) -> bool:
        """加载FAISS索引和节点映射"""
        try:
            if not os.path.exists(self.index_file) or not os.path.exists(self.mapping_file):
                print("索引文件不存在")
                return False
            
            # 加载FAISS索引
            self.index = faiss.read_index(self.index_file)
            print(f"FAISS索引已加载，共 {self.index.ntotal} 个向量")
            
            # 加载节点映射
            with open(self.mapping_file, 'rb') as f:
                mapping_data = pickle.load(f)
                self.node_mapping = mapping_data['node_mapping']
                self.reverse_mapping = mapping_data['reverse_mapping']
            
            print(f"节点映射已加载，共 {len(self.node_mapping)} 个节点")
            return True
            
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def search_similar_nodes(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        搜索相似节点
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相似节点列表
        """
        if self.index is None:
            print("FAISS索引未加载")
            return []
        
        try:
            # 生成查询向量
            query_vector = self.model.encode([query_text], convert_to_tensor=True)
            query_vector = query_vector.cpu().numpy()
            
            # 归一化查询向量
            faiss.normalize_L2(query_vector)
            
            # 搜索相似向量
            similarities, indices = self.index.search(query_vector.astype('float32'), top_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示没有结果
                    continue
                
                node_info = self.reverse_mapping.get(idx)
                if node_info is None:
                    continue
                
                results.append({
                    'node_id': node_info['id'],
                    'node_type': node_info['type'],
                    'node_name': node_info['name'],
                    'similarity': float(similarity),
                    'rank': i + 1
                })
            
            return results
            
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def search_by_type(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        搜索相似节点（保持向后兼容的接口）
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相似节点列表
        """
        return self.search_similar_nodes(query_text, top_k)
    
    def get_index_statistics(self) -> Dict:
        """
        获取索引统计信息
        
        Returns:
            统计信息
        """
        if self.index is None:
            return {"total_nodes": 0, "vector_dimension": 0, "node_types": {}}
        
        # 统计各类型节点数量
        type_counts = {}
        for node_info in self.reverse_mapping.values():
            node_type = node_info['type']
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        return {
            "total_nodes": self.index.ntotal,
            "vector_dimension": self.vector_dim,
            "node_types": type_counts
        }
    
    def update_index(self, graph_db, new_nodes_only: bool = True) -> bool:
        """
        更新索引
        
        Args:
            graph_db: 图数据库连接
            new_nodes_only: 是否只添加新节点
            
        Returns:
            是否更新成功
        """
        print("=== 更新FAISS索引 ===")
        
        try:
            if new_nodes_only and self.index is not None:
                # 只添加新节点
                existing_node_ids = set(self.node_mapping.keys())
                new_nodes = []
                
                for node_type in self.node_types:
                    query = f"""
                    MATCH (n:{node_type})
                    RETURN n.name as name
                    """
                    
                    results = graph_db.run(query).data()
                    
                    for result in results:
                        node_name = result['name']
                        node_id = f"{node_type}:{node_name}"
                        
                        if node_id not in existing_node_ids:
                            text = self.get_node_text(node_type, node_name)
                            new_nodes.append({
                                'id': node_id,
                                'type': node_type,
                                'name': node_name,
                                'text': text
                            })
                
                if not new_nodes:
                    print("没有新节点需要添加")
                    return True
                
                print(f"找到 {len(new_nodes)} 个新节点")
                
                # 为新节点生成向量
                node_texts = [node['text'] for node in new_nodes]
                embeddings = self.model.encode(node_texts, convert_to_tensor=True)
                embeddings = embeddings.cpu().numpy()
                
                # 归一化向量
                faiss.normalize_L2(embeddings)
                
                # 添加到索引
                start_idx = self.index.ntotal
                self.index.add(embeddings.astype('float32'))
                
                # 更新映射
                for i, node in enumerate(new_nodes):
                    idx = start_idx + i
                    self.node_mapping[node['id']] = idx
                    self.reverse_mapping[idx] = node
                
                print(f"成功添加 {len(new_nodes)} 个新节点")
                
            else:
                # 重新构建整个索引
                return self.build_faiss_index(graph_db)
            
            # 保存更新后的索引
            self.save_index()
            return True
            
        except Exception as e:
            print(f"更新索引失败: {e}")
            return False


def main():
    """主函数 - 用于测试FAISS向量构建功能"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config.config import Config
    from core.graph_database import GraphDatabase
    
    # 初始化配置和数据库连接
    config = Config()
    graph_db = GraphDatabase(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    # 初始化FAISS向量构建器
    vector_builder = FAISSVectorBuilder()
    
    # 构建FAISS索引
    if vector_builder.build_faiss_index(graph_db.graph):
        print("FAISS索引构建成功")
        
        # 显示统计信息
        stats = vector_builder.get_index_statistics()
        print("\n=== FAISS索引统计信息 ===")
        print(f"总节点数: {stats['total_nodes']}")
        print(f"向量维度: {stats['vector_dimension']}")
        print("各类型节点数量:")
        for node_type, count in stats['node_types'].items():
            print(f"  {node_type}: {count}")
        
        # 测试搜索功能
        print("\n=== 测试FAISS搜索功能 ===")
        test_queries = [
            "藿香正气水",
            "头痛",
            "柴胡",
            "伤寒论"
        ]
        
        for query_text in test_queries:
            print(f"\n搜索 '{query_text}':")
            results = vector_builder.search_by_type(query_text, top_k=5)
            
            if results:
                for result in results:
                    print(f"  {result['node_name']} (相似度: {result['similarity']:.3f}, 排名: {result['rank']})")
            else:
                print("  未找到相关结果")
    else:
        print("FAISS索引构建失败")


if __name__ == "__main__":
    main() 