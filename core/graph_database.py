from py2neo import Graph, Node, Relationship
import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict


class GraphDatabase:
    """图数据库管理类，整合图谱构建和查询功能"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", database="neo4j"):
        """
        初始化图数据库连接
        
        Args:
            uri: Neo4j连接地址
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.graph = Graph(uri, auth=(user, password), name=database)
        self.node_cache = {}  # 缓存已创建的节点，避免重复创建
        
    def clear_database(self):
        """清空数据库"""
        self.graph.run("MATCH (n) DETACH DELETE n")
        self.node_cache.clear()
        print("数据库已清空")
    
    def create_node(self, label: str, name: str, properties: Optional[Dict] = None) -> Node:
        """
        创建或获取节点
        
        Args:
            label: 节点标签
            name: 节点名称
            properties: 节点属性
            
        Returns:
            Node对象
        """
        node_key = f"{label}:{name}"
        
        if node_key in self.node_cache:
            return self.node_cache[node_key]
        
        # 创建节点
        if properties:
            node = Node(label, name=name, **properties)
        else:
            node = Node(label, name=name)
        
        self.graph.create(node)
        self.node_cache[node_key] = node
        return node
    
    def create_relationship(self, node1: Node, relation_type: str, node2: Node, 
                          properties: Optional[Dict] = None) -> Relationship:
        """
        创建关系
        
        Args:
            node1: 起始节点
            relation_type: 关系类型
            node2: 目标节点
            properties: 关系属性
            
        Returns:
            Relationship对象
        """
        if properties:
            relationship = Relationship(node1, relation_type, node2, **properties)
        else:
            relationship = Relationship(node1, relation_type, node2)
        
        self.graph.create(relationship)
        return relationship
    
    def build_from_json(self, json_file: str):
        """
        从JSON文件构建知识图谱
        
        Args:
            json_file: JSON文件路径
        """
        if not os.path.exists(json_file):
            print(f"文件不存在: {json_file}")
            return
        
        with open(json_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        
        print(f"开始构建知识图谱，共{len(triples)}个三元组...")
        
        for i, triple in enumerate(triples):
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(triples)}")
            
            try:
                # 解析节点信息
                node1_info = triple["node_1"].split("\t")
                node2_info = triple["node_2"].split("\t")
                
                if len(node1_info) != 2 or len(node2_info) != 2:
                    continue
                
                label1, name1 = node1_info
                label2, name2 = node2_info
                relation = triple["relation"]
                
                # 创建节点
                node1 = self.create_node(label1, name1)
                node2 = self.create_node(label2, name2)
                
                # 创建关系（支持权重属性）
                properties = {}
                if "weight" in triple:
                    properties["weight"] = triple["weight"]
                
                self.create_relationship(node1, relation, node2, properties)
                
            except Exception as e:
                print(f"处理三元组时出错: {e}")
                continue
        
        print("知识图谱构建完成")

    def search_formulas_by_symptoms(self, symptoms: List[str], limit: int = 10) -> List[Dict]:
        """
        根据症状搜索相关方剂（增强版，返回完整信息）

        Args:
            symptoms: 症状列表
            limit: 返回结果数量限制

        Returns:
            相关方剂列表，包含完整信息
        """
        try:
            # 构建症状匹配条件
            symptoms_conditions = []
            for symptom in symptoms:
                if symptom.strip():
                    symptoms_conditions.append(f"func.name CONTAINS '{symptom.strip()}'")

            if not symptoms_conditions:
                return []

            # 使用OR连接多个症状条件
            symptoms_where = " OR ".join(symptoms_conditions)

            # 修复Cypher查询，移除中文注释
            cypher_query = f"""
                MATCH (func:功能主治)
                WHERE {symptoms_where}
                OPTIONAL MATCH (formula:方名)-[:functions]->(func)
                WITH DISTINCT formula
                OPTIONAL MATCH (formula)-[:`prescription type`]->(prescription:处方)
                OPTIONAL MATCH (prescription)-[comp:composition]->(herb:中药名)
                OPTIONAL MATCH (formula)-[:functions]->(all_func:功能主治)
                OPTIONAL MATCH (formula)-[:from]->(source:来源)
                OPTIONAL MATCH (formula)-[:`another name`]->(alias:别名)
                RETURN DISTINCT
                       formula.name as formula_name,
                       collect(DISTINCT all_func.name) as functions,
                       collect(DISTINCT {{herb: herb.name, dose: comp.weight}}) as herbs,
                       collect(DISTINCT source.name) as sources,
                       collect(DISTINCT alias.name) as aliases
                LIMIT {limit}
            """

            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]

        except Exception as e:
            print(f"按症状搜索方剂出错: {e}")
            # 使用更简单的查询作为备用
            try:
                simple_query = f"""
                    MATCH (func:功能主治)
                    WHERE func.name CONTAINS '{symptoms[0] if symptoms else ""}'
                    MATCH (formula:方名)-[:functions]->(func)
                    RETURN DISTINCT formula.name as formula_name
                    LIMIT {limit}
                """
                result = self.graph.run(simple_query)
                formulas = [{"formula_name": record["formula_name"]} for record in result]
                return formulas
            except Exception as e2:
                print(f"备用查询也失败: {e2}")
                return []
    
    def get_formula_details(self, formula_name: str) -> Dict:
        """
        获取方剂详细信息
        
        Args:
            formula_name: 方剂名称
            
        Returns:
            方剂详细信息
        """
        cypher_query = f"""
        MATCH (formula:方名 {{name: '{formula_name}'}})
        OPTIONAL MATCH (formula)-[:from]->(source:来源)
        OPTIONAL MATCH (formula)-[:`another name`]->(alias:别名)
        OPTIONAL MATCH (formula)-[:functions]->(function:功能主治)
        OPTIONAL MATCH (formula)-[:`prescription type`]->(prescription:处方)
        OPTIONAL MATCH (prescription)-[comp:composition]->(herb:中药名)
        RETURN formula.name as formula_name,
               collect(DISTINCT source.name) as sources,
               collect(DISTINCT alias.name) as aliases,
               collect(DISTINCT function.name) as functions,
               collect(DISTINCT {{herb: herb.name, dose: comp.weight}}) as herbs
        """
        
        try:
            result = self.graph.run(cypher_query)
            record = result.data()[0] if result.data() else {}
            return record
        except Exception as e:
            print(f"查询方剂详情出错: {e}")
            return {}
    
    def get_related_entities(self, entity_name: str, entity_type: str, limit: int = 10) -> List[Dict]:
        """
        获取相关实体
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型
            limit: 返回结果数量限制
            
        Returns:
            相关实体列表
        """
        cypher_query = f"""
        MATCH (entity:{entity_type} {{name: '{entity_name}'}})-[r]-(related)
        RETURN type(r) as relation_type, 
               labels(related)[0] as related_type,
               related.name as related_name
        LIMIT {limit}
        """
        
        try:
            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]
        except Exception as e:
            print(f"查询相关实体出错: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict:
        """
        获取图谱统计信息
        
        Returns:
            图谱统计信息
        """
        stats_query = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY count DESC
        """
        
        try:
            result = self.graph.run(stats_query)
            node_stats = [dict(record) for record in result]
            
            # 获取关系统计
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as relation_type, count(r) as count
            ORDER BY count DESC
            """
            
            rel_result = self.graph.run(rel_query)
            rel_stats = [dict(record) for record in rel_result]
            
            return {
                "node_statistics": node_stats,
                "relationship_statistics": rel_stats
            }
        except Exception as e:
            print(f"获取统计信息出错: {e}")
            return {}
    
    def execute_query(self, cypher_query: str) -> List[Dict]:
        """
        执行Cypher查询语句
        
        Args:
            cypher_query: Cypher查询语句
            
        Returns:
            查询结果列表
        """
        try:
            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]
        except Exception as e:
            print(f"执行查询出错: {e}")
            return []
    
    def get_formula_with_weights(self, formula_name: str) -> Dict:
        """获取方剂详细信息，包括中药权重"""
        cypher_query = f"""
        MATCH (formula:方名 {{name: '{formula_name}'}})
        OPTIONAL MATCH (formula)-[:`prescription type`]->(prescription:处方)
        OPTIONAL MATCH (prescription)-[r:composition]->(herb:中药名)
        OPTIONAL MATCH (formula)-[:functions]->(function:功能主治)
        OPTIONAL MATCH (formula)-[:from]->(source:来源)
        OPTIONAL MATCH (formula)-[:`another name`]->(alias:别名)
        RETURN formula.name as formula_name,
               collect(DISTINCT function.name) as functions,
               collect(DISTINCT {{herb: herb.name, dose: r.weight}}) as herbs,
               collect(DISTINCT source.name) as sources,
               collect(DISTINCT alias.name) as aliases
        """
        
        try:
            result = self.graph.run(cypher_query)
            
            # 处理结果
            if result.data():
                record = result.data()[0]
                return {
                    "formula_name": record.get("formula_name"),
                    "functions": record.get("functions", []),
                    "herbs": record.get("herbs", []),
                    "sources": record.get("sources", []),
                    "aliases": record.get("aliases", [])
                }
            else:
                return {}
        except Exception as e:
            print(f"查询出错: {e}")
            return {}
    
    def search_formulas_by_name(self, name: str, limit: int = 10) -> List[Dict]:
        """
        根据方剂名称搜索相关方剂（支持模糊匹配）
        
        Args:
            name: 方剂名称（支持部分匹配）
            limit: 返回结果数量限制
            
        Returns:
            相关方剂列表，包含基本信息
        """
        try:
            # 使用CONTAINS进行模糊匹配，支持部分名称搜索
            cypher_query = f"""
            MATCH (formula:方名)
            WHERE formula.name CONTAINS '{name}' OR formula.name = '{name}'
            OPTIONAL MATCH (formula)-[:functions]->(func:功能主治)
            OPTIONAL MATCH (formula)-[:`prescription type`]->(prescription:处方)
            OPTIONAL MATCH (prescription)-[comp:composition]->(herb:中药名)
            OPTIONAL MATCH (formula)-[:from]->(source:来源)
            OPTIONAL MATCH (formula)-[:`another name`]->(alias:别名)
            RETURN DISTINCT
                   formula.name as formula_name,
                   collect(DISTINCT func.name) as functions,
                   collect(DISTINCT {{herb: herb.name, dose: comp.weight}}) as herbs,
                   collect(DISTINCT source.name) as sources,
                   collect(DISTINCT alias.name) as aliases
            ORDER BY 
                   CASE WHEN formula.name = '{name}' THEN 0 ELSE 1 END,  -- 完全匹配优先
                   length(formula.name)  -- 名称短的优先
            LIMIT {limit}
            """
            
            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]
            
        except Exception as e:
            print(f"按名称搜索方剂出错: {e}")
            return []
    
    def search_formulas_by_ingredient(self, herb: str, limit: int = 10) -> List[Dict]:
        """
        根据中药成分搜索包含该成分的方剂
        
        Args:
            herb: 中药名称（支持模糊匹配）
            limit: 返回结果数量限制
            
        Returns:
            包含该成分的方剂列表
        """
        try:
            # 先找到匹配的中药，再找到包含该中药的方剂
            cypher_query = f"""
            MATCH (herb:中药名)
            WHERE herb.name CONTAINS '{herb}' OR herb.name = '{herb}'
            OPTIONAL MATCH (prescription:处方)-[comp:composition]->(herb)
            OPTIONAL MATCH (formula:方名)-[:`prescription type`]->(prescription)
            OPTIONAL MATCH (formula)-[:functions]->(func:功能主治)
            OPTIONAL MATCH (formula)-[:from]->(source:来源)
            OPTIONAL MATCH (formula)-[:`another name`]->(alias:别名)
            WITH DISTINCT formula, herb, comp
            OPTIONAL MATCH (formula)-[:`prescription type`]->(prescription2:处方)
            OPTIONAL MATCH (prescription2)-[comp2:composition]->(herb2:中药名)
            RETURN DISTINCT
                   formula.name as formula_name,
                   collect(DISTINCT func.name) as functions,
                   collect(DISTINCT {{herb: herb2.name, dose: comp2.weight}}) as herbs,
                   collect(DISTINCT source.name) as sources,
                   collect(DISTINCT alias.name) as aliases,
                   herb.name as search_herb,  -- 记录搜索的中药
                   comp.weight as herb_dose  -- 记录该中药在方剂中的剂量
            ORDER BY 
                   CASE WHEN herb.name = '{herb}' THEN 0 ELSE 1 END,  -- 完全匹配优先
                   comp.weight DESC  -- 剂量大的优先（可能更重要）
            LIMIT {limit}
            """
            
            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]
            
        except Exception as e:
            print(f"按成分搜索方剂出错: {e}")
            return []
    
    def search_formulas_by_function(self, function_desc: str, limit: int = 10) -> List[Dict]:
        """
        根据功能主治描述搜索相关方剂
        
        Args:
            function_desc: 功能描述（如：清热解毒、补气养血等）
            limit: 返回结果数量限制
            
        Returns:
            相关方剂列表
        """
        try:
            cypher_query = f"""
            MATCH (func:功能主治)
            WHERE func.name CONTAINS '{function_desc}'
            OPTIONAL MATCH (formula:方名)-[:functions]->(func)
            WITH DISTINCT formula
            OPTIONAL MATCH (formula)-[:`prescription type`]->(prescription:处方)
            OPTIONAL MATCH (prescription)-[comp:composition]->(herb:中药名)
            OPTIONAL MATCH (formula)-[:functions]->(all_func:功能主治)
            OPTIONAL MATCH (formula)-[:from]->(source:来源)
            OPTIONAL MATCH (formula)-[:`another name`]->(alias:别名)
            RETURN DISTINCT
                   formula.name as formula_name,
                   collect(DISTINCT all_func.name) as functions,
                   collect(DISTINCT {{herb: herb.name, dose: comp.weight}}) as herbs,
                   collect(DISTINCT source.name) as sources,
                   collect(DISTINCT alias.name) as aliases
            ORDER BY 
                   CASE WHEN func.name = '{function_desc}' THEN 0 ELSE 1 END,  -- 完全匹配优先
                   length(formula.name)  -- 名称短的优先
            LIMIT {limit}
            """
            
            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]
            
        except Exception as e:
            print(f"按功能搜索方剂出错: {e}")
            return []
    
    def get_formula_ingredients(self, formula_name: str) -> Dict:
        """
        获取方剂的详细组成信息
        
        Args:
            formula_name: 方剂名称
            
        Returns:
            方剂组成信息
        """
        try:
            cypher_query = f"""
            MATCH (formula:方名 {{name: '{formula_name}'}})
            OPTIONAL MATCH (formula)-[:`prescription type`]->(prescription:处方)
            OPTIONAL MATCH (prescription)-[comp:composition]->(herb:中药名)
            OPTIONAL MATCH (formula)-[:functions]->(func:功能主治)
            OPTIONAL MATCH (formula)-[:from]->(source:来源)
            OPTIONAL MATCH (formula)-[:`another name`]->(alias:别名)
            RETURN DISTINCT
                   formula.name as formula_name,
                   collect(DISTINCT func.name) as functions,
                   collect(DISTINCT {{herb: herb.name, dose: comp.weight}}) as herbs,
                   collect(DISTINCT source.name) as sources,
                   collect(DISTINCT alias.name) as aliases
            """
            
            result = self.graph.run(cypher_query)
            if result.data():
                record = result.data()[0]
                return {
                    "formula_name": record.get("formula_name"),
                    "functions": record.get("functions", []),
                    "herbs": record.get("herbs", []),
                    "sources": record.get("sources", []),
                    "aliases": record.get("aliases", [])
                }
            else:
                return {}
                
        except Exception as e:
            print(f"获取方剂组成出错: {e}")
            return {}