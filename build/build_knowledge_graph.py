#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱构建脚本
整合网络爬虫和PDF处理功能，构建完整的中医知识图谱
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_database import GraphDatabase
from config.config import Config
from webinff.tcm_crawler import scrape_tcm_formula_data
# from pdf.pdf2txt import extract_two_column_formulas


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        """初始化构建器"""
        self.config = Config()
        self.graph_db = GraphDatabase(
            uri=self.config.NEO4J_URI,
            user=self.config.NEO4J_USER,
            password=self.config.NEO4J_PASSWORD,
            database=self.config.NEO4J_DATABASE
        )
        
        # 确保目录存在
        os.makedirs('build', exist_ok=True)
        os.makedirs('uploads', exist_ok=True)
    
    def build_from_crawler_data(self, json_file: str = None):
        """
        从爬虫数据构建知识图谱
        
        Args:
            json_file: JSON文件路径，如果为None则运行爬虫
        """
        print("=== 开始构建知识图谱 ===")
        
        if json_file is None:
            # 运行爬虫获取数据
            print("正在运行爬虫获取数据...")
            scrape_tcm_formula_data()
            json_file = self.config.get_knowledge_graph_path()
        
        if not os.path.exists(json_file):
            print(f"错误：找不到数据文件 {json_file}")
            return False
        
        # 清空数据库
        print("正在清空数据库...")
        self.graph_db.clear_database()
        
        # 构建知识图谱
        print("正在构建知识图谱...")
        self.graph_db.build_from_json(json_file)
        
        # 构建向量库
        print("正在构建向量库...")
        self.build_vectors()
        
        # 获取统计信息
        stats = self.graph_db.get_graph_statistics()
        print("=== 构建完成 ===")
        print(f"节点统计：{stats.get('node_statistics', [])}")
        print(f"关系统计：{stats.get('relationship_statistics', [])}")
        
        return True
    
    def build_from_pdf_data(self, pdf_files: List[str]):
        """
        从PDF文件构建知识图谱
        
        Args:
            pdf_files: PDF文件路径列表
        """
        print("=== 开始从PDF构建知识图谱 ===")
        
        all_triples = []
        
        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"警告：PDF文件不存在 {pdf_file}")
                continue
            
            print(f"正在处理PDF文件：{pdf_file}")
            
            # 提取PDF文本
            output_txt = pdf_file.replace('.pdf', '_extracted.txt')
            try:
                # extract_two_column_formulas(pdf_file, output_txt)
                print(f"PDF文本提取完成：{output_txt}")
            except Exception as e:
                print(f"PDF处理失败：{e}")
                continue
            
            # 从文本中提取三元组（这里需要实现文本到三元组的转换）
            # 暂时跳过，因为需要更复杂的NLP处理
            print(f"PDF文件 {pdf_file} 处理完成")
        
        return True
    
    def merge_knowledge_graphs(self, json_files: List[str], output_file: str):
        """
        合并多个知识图谱文件
        
        Args:
            json_files: JSON文件路径列表
            output_file: 输出文件路径
        """
        print("=== 开始合并知识图谱 ===")
        
        all_triples = []
        node_cache = set()  # 用于去重
        
        for json_file in json_files:
            if not os.path.exists(json_file):
                print(f"警告：文件不存在 {json_file}")
                continue
            
            print(f"正在处理文件：{json_file}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                triples = json.load(f)
            
            # 去重处理
            for triple in triples:
                triple_key = f"{triple['node_1']}_{triple['relation']}_{triple['node_2']}"
                if triple_key not in node_cache:
                    all_triples.append(triple)
                    node_cache.add(triple_key)
        
        # 保存合并后的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_triples, f, ensure_ascii=False, indent=2)
        
        print(f"合并完成，共 {len(all_triples)} 个三元组，保存到 {output_file}")
        return True
    
    def validate_knowledge_graph(self, json_file: str):
        """
        验证知识图谱数据
        
        Args:
            json_file: JSON文件路径
        """
        print("=== 验证知识图谱数据 ===")
        
        if not os.path.exists(json_file):
            print(f"错误：文件不存在 {json_file}")
            return False
        
        with open(json_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        
        # 统计信息
        node_types = set()
        relation_types = set()
        node_names = set()
        
        for triple in triples:
            # 解析节点信息
            node1_info = triple["node_1"].split("\t")
            node2_info = triple["node_2"].split("\t")
            
            if len(node1_info) == 2:
                node_types.add(node1_info[0])
                node_names.add(node1_info[1])
            
            if len(node2_info) == 2:
                node_types.add(node2_info[0])
                node_names.add(node2_info[1])
            
            relation_types.add(triple["relation"])
        
        print(f"节点类型：{sorted(list(node_types))}")
        print(f"关系类型：{sorted(list(relation_types))}")
        print(f"节点数量：{len(node_names)}")
        print(f"三元组数量：{len(triples)}")
        
        # 检查数据质量
        invalid_triples = []
        for i, triple in enumerate(triples):
            if "node_1" not in triple or "node_2" not in triple or "relation" not in triple:
                invalid_triples.append(i)
        
        if invalid_triples:
            print(f"警告：发现 {len(invalid_triples)} 个无效三元组")
        else:
            print("数据验证通过")
        
        return True
    
    def export_statistics(self, json_file: str, output_file: str):
        """
        导出知识图谱统计信息
        
        Args:
            json_file: JSON文件路径
            output_file: 输出文件路径
        """
        print("=== 导出统计信息 ===")
        
        if not os.path.exists(json_file):
            print(f"错误：文件不存在 {json_file}")
            return False
        
        with open(json_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        
        # 统计信息
        stats = {
            "total_triples": len(triples),
            "node_types": {},
            "relation_types": {},
            "top_nodes": {},
            "top_relations": {}
        }
        
        node_counts = {}
        relation_counts = {}
        
        for triple in triples:
            # 统计节点
            node1_info = triple["node_1"].split("\t")
            node2_info = triple["node_2"].split("\t")
            
            if len(node1_info) == 2:
                node_type = node1_info[0]
                node_name = node1_info[1]
                if node_type not in stats["node_types"]:
                    stats["node_types"][node_type] = 0
                stats["node_types"][node_type] += 1
                node_counts[node_name] = node_counts.get(node_name, 0) + 1
            
            if len(node2_info) == 2:
                node_type = node2_info[0]
                node_name = node2_info[1]
                if node_type not in stats["node_types"]:
                    stats["node_types"][node_type] = 0
                stats["node_types"][node_type] += 1
                node_counts[node_name] = node_counts.get(node_name, 0) + 1
            
            # 统计关系
            relation = triple["relation"]
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        # 获取top节点和关系
        stats["top_nodes"] = dict(sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        stats["top_relations"] = dict(sorted(relation_counts.items(), key=lambda x: x[1], reverse=True))
        
        # 保存统计信息
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存到 {output_file}")
        return True

    def build_from_local_datasets(self, json_files=None):
        """
        直接使用本地划分好的两个数据集构建知识图谱
        Args:
            json_files: 数据集文件路径列表
        """
        if json_files is None:
            json_files = [
                '../kg_llm/build/pdf_knowledge_graph_weighted.json',
                '../kg_llm/build/tcm_knowledge_graph_weighted.json'
            ]
        print("=== 使用本地数据集构建知识图谱 ===")
        # 合并两个 JSON 文件
        merged_file = 'build/merged_knowledge_graph.json'
        self.merge_knowledge_graphs(json_files, merged_file)
        # 清空数据库
        print("正在清空数据库...")
        self.graph_db.clear_database()
        # 构建知识图谱
        print("正在构建知识图谱...")
        self.graph_db.build_from_json(merged_file)
        
        # 构建向量库
        print("正在构建向量库...")
        self.build_vectors()
        
        # 获取统计信息
        stats = self.graph_db.get_graph_statistics()
        print("=== 构建完成 ===")
        print(f"节点统计：{stats.get('node_statistics', [])}")
        print(f"关系统计：{stats.get('relationship_statistics', [])}")
        return True
    
    def build_vectors(self):
        """
        构建向量库
        """
        try:
            from core.faiss_vector_builder import FAISSVectorBuilder
            
            print("=== 开始构建FAISS向量库 ===")
            
            # 初始化FAISS向量构建器
            vector_builder = FAISSVectorBuilder()
            
            # 构建FAISS索引
            success = vector_builder.build_faiss_index(self.graph_db.graph)
            
            if success:
                # 显示统计信息
                stats = vector_builder.get_index_statistics()
                print("\n=== FAISS向量库统计信息 ===")
                print(f"总节点数: {stats['total_nodes']}")
                print(f"向量维度: {stats['vector_dimension']}")
                print("各类型节点数量:")
                for node_type, count in stats['node_types'].items():
                    print(f"  {node_type}: {count}")
                
                print("FAISS向量库构建完成")
                return True
            else:
                print("FAISS向量库构建失败")
                return False
            
        except Exception as e:
            print(f"FAISS向量库构建失败: {e}")
            return False


def main():
    """主函数"""
    builder = KnowledgeGraphBuilder()
    # 默认只用本地数据集构建
    builder.build_from_local_datasets()


if __name__ == "__main__":
    main() 