from .aliyun_qwen_api import get_local_model
import time
import json
from typing import List, Dict, Optional, Any, Tuple
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .graph_database import GraphDatabase
from .multimodal_processor import MultimodalProcessor
#from .local_model import get_local_model


class QueryAnalyzer:
    """问题分析器 - 第一步：问题改写并提取关键词，判断是否需要检索"""

    def __init__(self, api_key: str = None):
        self.local_model = get_local_model()

    def analyze_query(self, query: str, history: str = "") -> Dict:
        """
        分析用户问题，改写并提取关键词，判断是否需要检索

        Args:
            query: 原始问题
            history: 对话历史

        Returns:
            {
                "need_retrieval": bool,  # 是否需要检索
                "rewritten_query": str,  # 改写后的问题
                "search_type": str,      # 检索类型 (1:方剂名, 2:成分, 3:疾病, 0:无关)
                "keywords": List[str],   # 提取的关键词
                "reason": str           # 判断原因
            }
        """
        prompt = f"""
        你是一个专业的中医问答助手。请分析用户的原始问题，判断是否可以通过检索中医方剂知识库来回答。

        对话历史：{history}
        原始问题：{query}

        请按以下格式返回JSON结果：
        {{
            "need_retrieval": true/false,
            "rewritten_query": "改写后的问题",
            "search_type": "检索类型标识",
            "keywords": ["关键词1", "关键词2"],
            "reason": "判断原因"
        }}

        检索类型标识说明：
        - "1": 通过方剂名称检索方剂 (如：藿香正气水、六味地黄丸)
        - "2": 通过中药成分检索方剂 (如：柴胡、当归、人参)
        - "3": 通过疾病症状或者功能检索方剂 (如：头痛、感冒、失眠、清热解毒)
        - "0": 无关问题 (如：医学常识、其他问题)

        判断标准：
        1. 如果问题涉及具体的中药方剂、中药成分、疾病症状，需要检索
        2. 如果问题是一般性医学常识或与中医无关，不需要检索
        3. 关键词应该是具体的药名、成分名或症状名(也可以是你诊断的疾病，比如：气虚血瘀证等)，判断用户的问题与对话历史是否有关联，以此构建关键词

        请确保返回的是有效的JSON格式，除此之外不要任何多余内容。
        """
        try:
            response = self.local_model.call(
                messages=[
                    {"role": "system", "content": "你是一个专业的中医问题分析助手，请严格按照JSON格式返回结果"},
                    {"role": "user", "content": prompt}
                ],
                enable_thinking=False  # 禁用思考功能，避免生成<think>标签
            )
            if response["status_code"] == 200:
                result_text = response["output"]["choices"][0]["message"]["content"]
                # 尝试解析JSON
                try:
                    result = json.loads(result_text)
                    return {
                        "need_retrieval": result.get("need_retrieval", False),
                        "rewritten_query": result.get("rewritten_query", query),
                        "search_type": result.get("search_type", "0"),
                        "keywords": result.get("keywords", []),
                        "reason": result.get("reason", "无法判断")
                    }
                except json.JSONDecodeError:
                    # 如果JSON解析失败，返回默认结果
                    return {
                        "need_retrieval": False,
                        "rewritten_query": query,
                        "search_type": "0",
                        "keywords": [],
                        "reason": "JSON解析失败"+result_text
                    }
            else:
                return {
                    "need_retrieval": False,
                    "rewritten_query": query,
                    "search_type": "0",
                    "keywords": [],
                    "reason": "模型调用失败"
                }

        except Exception as e:
            print(f"问题分析失败: {e}")
            return {
                "need_retrieval": False,
                "rewritten_query": query,
                "search_type": "0",
                "keywords": [],
                "reason": f"分析出错: {e}"
            }


class VectorRetriever:
    """
    向量检索器 - 第二步：向量检索相关节点

    设计理念：
    - 向量检索的目的是精确匹配节点名称
    - 不按节点类型过滤，返回所有相似节点
    - 让大模型根据检索结果进行智能分析和过滤
    """

    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db
        # 初始化FAISS向量构建器
        try:
            from .faiss_vector_builder import FAISSVectorBuilder
            self.vector_builder = FAISSVectorBuilder()
            self.faiss_loaded = self.vector_builder.load_index()
            if not self.faiss_loaded:
                print("FAISS索引加载失败，将使用传统检索方法")
        except Exception as e:
            print(f"FAISS向量构建器初始化失败: {e}")
            self.vector_builder = None
            self.faiss_loaded = False

    def vector_search(self, keywords: List[str], search_type: str, similarity_threshold: float = 0.6, max_per_keyword: int = 20) -> List[Dict]:
        """
        根据关键词进行向量检索（基于相似度阈值筛选）

        Args:
            keywords: 关键词列表
            search_type: 检索类型（保留参数以兼容接口）
            similarity_threshold: 相似度阈值（0~1）
            max_per_keyword: 每个关键词从索引中初取的最大候选数（随后按阈值过滤）

        Returns:
            检索到的相关节点列表（去重、按相似度降序）
        """
        try:
            if not self.vector_builder or not self.faiss_loaded:
                print("FAISS索引不可用，使用传统检索方法")
                return self._fallback_search(keywords, search_type, max_per_keyword)

            # 使用FAISS向量搜索（不按类型过滤），先取较多，再按阈值过滤
            all_results = []
            for keyword in keywords:
                similar_nodes = self.vector_builder.search_by_type(
                    keyword, max_per_keyword
                )
                # 追加来源关键词，便于后续构建查询按关键词分组（可选）
                for n in similar_nodes:
                    n["_source_keyword"] = keyword
                all_results.extend(similar_nodes)

            # 基于阈值过滤
            filtered = [r for r in all_results if r.get("similarity", 0) >= similarity_threshold]

            # 去重（保留最高相似度）
            unique_results = {}
            for result in filtered:
                node_id = result["node_id"]
                if node_id not in unique_results or result["similarity"] > unique_results[node_id]["similarity"]:
                    unique_results[node_id] = result

            # 转换为标准格式并排序
            final_results = []
            for result in sorted(unique_results.values(), key=lambda x: x["similarity"], reverse=True):
                final_results.append({
                    "name": result["node_name"],
                    "type": result["node_type"],
                    "similarity": result["similarity"],
                    "_source_keyword": result.get("_source_keyword")
                })

            return final_results

        except Exception as e:
            print(f"FAISS向量检索失败: {e}")
            return self._fallback_search(keywords, search_type, max_per_keyword)

    def _fallback_search(self, keywords: List[str], search_type: str, limit: int) -> List[Dict]:
        """传统检索方法（备用）"""
        try:
            if search_type == "1":  # 方剂名称检索
                results = []
                for keyword in keywords:
                    formulas = self.graph_db.search_formulas_by_name(keyword, limit)
                    results.extend(formulas)
                return results[:limit]
            elif search_type == "2":  # 成分检索
                results = []
                for keyword in keywords:
                    formulas = self.graph_db.search_formulas_by_ingredient(keyword, limit)
                    results.extend(formulas)
                return results[:limit]
            elif search_type == "3":  # 疾病检索
                results = []
                for keyword in keywords:
                    formulas = self.graph_db.search_formulas_by_symptoms([keyword], limit)
                    results.extend(formulas)
                return results[:limit]
            else:
                return []
        except Exception as e:
            print(f"传统检索失败: {e}")
            return []


class QueryBuilder:
    """查询构建器 - 第四步：根据检索结果构建查询语句"""

    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db

    def build_queries(self, retrieved_nodes: List[Dict], search_type: str) -> List[str]:
        """
        根据检索到的节点和搜索类型构建查询语句

        Args:
            retrieved_nodes: 检索到的节点
            search_type: 搜索类型

        Returns:
            Cypher查询语句列表
        """
        queries = []

        for node in retrieved_nodes:
            if search_type == "1":  # 方剂名称
                # 查询方剂的详细信息，按照正确的图数据库结构
                formula_name = node.get("name", "")
                if formula_name:
                    query = f"""
                    MATCH (formula:方名 {{name: '{formula_name}'}})
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
                    """
                    queries.append(query)

            elif search_type == "2":  # 成分
                # 查询包含该成分的方剂，按照正确的图数据库结构
                ingredient = node.get("name", "")
                if ingredient:
                    query = f"""
                    MATCH (herb:中药名 {{name: '{ingredient}'}})
                    OPTIONAL MATCH (prescription:处方)-[comp:composition]->(herb)
                    OPTIONAL MATCH (formula:方名)-[:`prescription type`]->(prescription)
                    OPTIONAL MATCH (formula)-[:functions]->(func:功能主治)
                    RETURN DISTINCT
                           formula.name as formula_name,
                           collect(DISTINCT func.name) as functions,
                           collect(DISTINCT {{herb: herb.name, dose: comp.weight}}) as herbs
                    """
                    queries.append(query)

            elif search_type == "3":  # 疾病
                # 查询治疗该症状的方剂，按照正确的图数据库结构
                # 重要：这里要获取方剂的所有功能主治，而不仅仅是匹配的症状
                symptom = node.get("name", "")
                if symptom:
                    query = f"""
                    MATCH (func:功能主治)
                    WHERE func.name CONTAINS '{symptom}'
                    OPTIONAL MATCH (formula:方名)-[:functions]->(func)
                    WITH DISTINCT formula
                    OPTIONAL MATCH (formula)-[:`prescription type`]->(prescription:处方)
                    OPTIONAL MATCH (prescription)-[comp:composition]->(herb:中药名)
                    // 获取该方剂的所有功能主治
                    OPTIONAL MATCH (formula)-[:functions]->(all_func:功能主治)
                    RETURN DISTINCT
                           formula.name as formula_name,
                           collect(DISTINCT all_func.name) as functions,
                           collect(DISTINCT {{herb: herb.name, dose: comp.weight}}) as herbs
                    """
                    queries.append(query)

        return queries

class ResultVisualizer:
    """结果可视化器 - 第五步：可视化检索结果"""

    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db

        # 定义节点颜色映射
        self.color_map = {
            "方剂": "#C6EB87",
            "方名": "#87EBC1",
            "功能主治": "#EE90A1",
            "处方": "#FFD700",
            "中药名": "#69B2FF",
            "症状": "#FF6B6B",
            "疾病": "#4ECDC4",
            "证候": "#45B7D1"
        }

    def visualize_results(self, query_results: List[Dict]) -> Dict:
        """
        可视化查询结果，生成网络图数据

        Args:
            query_results: 查询结果列表

        Returns:
            包含节点和边的可视化数据
        """
        if not query_results:
            return {"nodes": [], "edges": []}

        try:
            # 收集节点和边
            nodes = {}
            edges = set()

            for i, result in enumerate(query_results):
                try:
                    if isinstance(result, dict):
                        self._extract_nodes_and_edges(result, nodes, edges)
                    else:
                        continue
                except Exception as e:
                    print(f"警告：处理结果 {i} 时出错: {e}")
                    print(f"结果内容: {result}")
                    continue

            # 转换为vis.js格式（增大节点与字体尺寸）
            vis_nodes = []
            vis_edges = []

            for node_id, node_data in nodes.items():
                is_main = node_data["type"] in ["方剂", "方名"]
                vis_nodes.append({
                    "id": node_id,
                    "label": node_data["name"],
                    "title": f"{node_data['type']}: {node_data['name']}",
                    "color": node_data["color"],
                    "size": 72 if node_data["type"] == "方剂" else (54 if is_main else 40),
                    "shape": "dot",
                    "font": {
                        "size": 48 if node_data["type"] == "方剂" else (36 if is_main else 24),
                        "face": "Microsoft YaHei",
                        "bold": is_main
                    }
                })

            for edge in edges:
                source_id, target_id, label = edge
                vis_edges.append({
                    "from": source_id,
                    "to": target_id,
                    "label": label,
                    "arrows": "to",
                    "width": 2,
                    "font": {
                        "size": 12,
                        "face": "Microsoft YaHei"
                    },
                    "color": "#666666"
                })

            return {
                "nodes": vis_nodes,
                "edges": vis_edges,
                "options": {
                    "physics": {
                        "stabilization": False,
                        "barnesHut": {
                            "gravitationalConstant": -20000,
                            "springConstant": 0.02,
                            "springLength": 100
                        }
                    },
                    "interaction": {
                        "navigationButtons": True,
                        "keyboard": True,
                        "hover": True
                    },
                    "layout": {
                        "improvedLayout": True,
                        "hierarchical": {
                            "enabled": False
                        }
                    }
                }
            }
        except Exception as e:
            print(f"可视化结果生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回空的可视化结果，避免整个流程失败
            return {"nodes": [], "edges": []}

    def _extract_nodes_and_edges(self, result: Dict, nodes: Dict, edges: set):
        """从查询结果中提取节点和边，按照实际的关系构建"""

        # 如果result包含formula_name，说明这是一个方剂查询结果
        if 'formula_name' in result and result['formula_name']:
            formula_name = result['formula_name']

            # 创建方名节点
            formula_id = f"formula_{formula_name}"
            nodes[formula_id] = {
                "name": formula_name,
                "type": "方名",
                "color": self.color_map["方名"]
            }

            # 创建方剂根节点，并连接所有方名
            if 'fangji_root' not in nodes:
                nodes['fangji_root'] = {
                    "name": "方剂",
                    "type": "方剂",
                    "color": self.color_map["方剂"]
                }
            edges.add(('fangji_root', formula_id, "包含"))

            # 方名 -> 功能主治
            if 'functions' in result and result['functions']:
                for func_name in result['functions']:
                    if func_name:
                        func_id = f"func_{func_name}"
                        nodes[func_id] = {
                            "name": func_name,
                            "type": "功能主治",
                            "color": self.color_map["功能主治"]
                        }
                        edges.add((formula_id, func_id, "功能主治"))

            # 方名 -> 处方 -> 中药名（中药节点不显示剂量）
            if 'herbs' in result and result['herbs']:
                # 创建处方节点
                prescription_id = f"prescription_{formula_name}"
                nodes[prescription_id] = {
                    "name": f"{formula_name}处方",
                    "type": "处方",
                    "color": self.color_map["处方"]
                }

                # 方名 -> 处方
                edges.add((formula_id, prescription_id, "配方"))

                # 处方 -> 中药名（不附加剂量在label中）
                for herb_info in result['herbs']:
                    if isinstance(herb_info, dict) and 'herb' in herb_info:
                        herb_name = herb_info['herb']
                        if herb_name:
                            herb_id = f"herb_{herb_name}"
                            nodes[herb_id] = {
                                "name": herb_name,
                                "type": "中药名",
                                "color": self.color_map["中药名"]
                            }
                            edges.add((prescription_id, herb_id, "中药组成"))

        # 兼容旧的查询结果格式（保持简化关系）
        else:
            # 处理Neo4j Node对象和普通字典
            for key, value in result.items():
                try:
                    if hasattr(value, 'labels') and hasattr(value, 'get'):
                        # Neo4j Node对象
                        try:
                            if hasattr(value, 'element_id'):
                                node_id = value.element_id
                            elif hasattr(value, 'id'):
                                node_id = str(value.id)
                            else:
                                node_id = f"node_{hash(str(value))}"
                        except Exception:
                            node_id = f"node_{hash(str(value))}"

                        node_type = list(value.labels)[0] if value.labels else key
                        node_name = value.get('name', '')

                        if node_name and node_type in self.color_map:
                            nodes[node_id] = {
                                "name": node_name,
                                "type": node_type,
                                "color": self.color_map[node_type]
                            }

                    elif isinstance(value, dict) and value.get('name'):
                        node_id = f"{key}_{value['name']}"
                        node_type = key
                        node_name = value['name']

                        if node_type in self.color_map:
                            nodes[node_id] = {
                                "name": node_name,
                                "type": node_type,
                                "color": self.color_map[node_type]
                            }

                except Exception as e:
                    print(f"处理节点时出错: key={key}, error={e}")
                    continue

            # 为旧格式构建简单的边关系（基于节点类型）
            formula_nodes = [(id, data) for id, data in nodes.items() if data["type"] == "方名"]
            func_nodes = [(id, data) for id, data in nodes.items() if data["type"] == "功能主治"]
            herb_nodes = [(id, data) for id, data in nodes.items() if data["type"] == "中药名"]

            # 加入方剂根节点并连接所有方名
            if formula_nodes:
                if 'fangji_root' not in nodes:
                    nodes['fangji_root'] = {
                        "name": "方剂",
                        "type": "方剂",
                        "color": self.color_map["方剂"]
                    }
                for formula_id, _ in formula_nodes:
                    edges.add(('fangji_root', formula_id, "包含"))

            # 方名 -> 功能主治
            for formula_id, _ in formula_nodes:
                for func_id, _ in func_nodes:
                    edges.add((formula_id, func_id, "功能主治"))

            # 方名 -> 中药名（简化关系）
            for formula_id, _ in formula_nodes:
                for herb_id, _ in herb_nodes:
                    edges.add((formula_id, herb_id, "包含"))


class OptimizedRAGEngine:
    """优化版RAG引擎 - 实现五步流程"""

    def __init__(self, graph_db: GraphDatabase, api_key: str = None):
        """
        初始化优化版RAG引擎

        Args:
            graph_db: 图数据库实例
            api_key: API密钥
        """
        self.graph_db = graph_db
        self.api_key = api_key

        # 初始化各个组件
        self.query_analyzer = QueryAnalyzer(api_key)
        self.vector_retriever = VectorRetriever(graph_db)
        self.query_builder = QueryBuilder(graph_db)
        self.result_visualizer = ResultVisualizer(graph_db)
        self.answer_generator = AnswerGenerator(api_key)

        # 多模态处理器
        self.multimodal_processor = MultimodalProcessor(api_key)

        # 对话记忆 - 增大记忆轮数，保存问题分析结果而不是问答对
        self.memory = ConversationBufferWindowMemory(k=20, return_messages=False)
        print(f"初始化记忆系统，记忆轮数: 20")

    def _summarize_formulas(self, query_results: List[Dict]) -> str:
        """
        总结药方信息，只提取药方名称和功能主治（不包含组成成分）

        Args:
            query_results: 查询结果列表

        Returns:
            格式化的药方总结信息
        """
        if not query_results:
            return "未找到相关药方信息"

        # 提取药方信息
        formula_info = {}

        for result in query_results:
            if isinstance(result, dict):
                # 新的查询结果格式
                if 'formula_name' in result and result['formula_name']:
                    formula_name = result['formula_name']

                    if formula_name not in formula_info:
                        formula_info[formula_name] = {
                            'functions': set()
                        }

                    # 处理功能主治
                    if 'functions' in result and result['functions']:
                        for func in result['functions']:
                            if func:
                                formula_info[formula_name]['functions'].add(func)

                # 兼容旧的查询结果格式
                else:
                    # 查找药方名称和功能主治
                    formula_name = None
                    herb_function = None

                    for key, value in result.items():
                        if hasattr(value, 'labels') and hasattr(value, 'get'):
                            # Neo4j Node对象
                            if '方名' in value.labels:
                                formula_name = value.get('name', '')
                            elif '功能主治' in value.labels:
                                herb_function = value.get('name', '')
                        elif isinstance(value, dict) and value.get('name'):
                            # 普通字典
                            if key == 'formula_name' or '方名' in str(key):
                                formula_name = value['name']
                            elif key == 'herb_function' or '功能' in str(key):
                                herb_function = value['name']

                    # 如果找到了药方信息，添加到总结中
                    if formula_name:
                        if formula_name not in formula_info:
                            formula_info[formula_name] = {
                                'functions': set()
                            }

                        if herb_function:
                            formula_info[formula_name]['functions'].add(herb_function)

        # 格式化输出
        if not formula_info:
            return "未找到相关药方信息"

        summary_lines = []
        for formula_name, info in formula_info.items():
            functions_str = "、".join(info['functions']) if info['functions'] else "未知"
            summary_lines.append(f"{formula_name}：{functions_str}")

        return "\n".join(summary_lines)

    def _clean_query_results(self, query_results: List[Dict]) -> List[Dict]:
        """
        清洗与合并查询结果：
        - 过滤方名缺失或为"未知药方/未知/unknown"等占位名的记录
        - 将相同方名的记录合并（functions去重合并，herbs按中药名合并，sources/aliases去重）
        """
        if not query_results:
            return []

        INVALID_NAMES = {"未知药方", "未知", "unknown", "Unknown", "UNK"}
        merged: Dict[str, Dict] = {}

        for item in query_results:
            if not isinstance(item, dict):
                continue
            formula_name = str(item.get("formula_name", "")).strip()
            if not formula_name or formula_name in INVALID_NAMES:
                continue

            functions = item.get("functions") or []
            herbs = item.get("herbs") or []
            sources = item.get("sources") or []
            aliases = item.get("aliases") or []

            if formula_name not in merged:
                merged[formula_name] = {
                    "formula_name": formula_name,
                    "functions": set(),
                    "herbs": {},  # herb_name -> dose
                    "sources": set(),
                    "aliases": set(),
                }

            bucket = merged[formula_name]
            for f in functions:
                if f:
                    bucket["functions"].add(f)
            for h in herbs:
                if isinstance(h, dict):
                    herb_name = str(h.get("herb", "")).strip()
                    dose = str(h.get("dose", "")).strip()
                    if herb_name:
                        # 若重复，以首次出现为准，或更新非空剂量
                        if herb_name not in bucket["herbs"] or (dose and not bucket["herbs"][herb_name]):
                            bucket["herbs"][herb_name] = dose
            for s in sources:
                if s:
                    bucket["sources"].add(s)
            for a in aliases:
                if a:
                    bucket["aliases"].add(a)

        cleaned_results: List[Dict] = []
        for fname, data in merged.items():
            herbs_list = [{"herb": hn, "dose": d} for hn, d in data["herbs"].items()]
            cleaned_results.append({
                "formula_name": fname,
                "functions": sorted(list(data["functions"])),
                "herbs": herbs_list,
                "sources": sorted(list(data["sources"])),
                "aliases": sorted(list(data["aliases"]))
            })

        return cleaned_results

    def _generate_table_data(self, query_results: List[Dict]) -> List[Dict]:
        """
        生成用于前端表格展示的药方数据

        Args:
            query_results: 查询结果列表

        Returns:
            格式化后的药方数据列表
        """
        table_data = []
        seen_formulas = set()  # 用于去重
        INVALID_NAMES = {"未知药方", "未知", "unknown", "Unknown", "UNK"}

        for result in query_results:
            if isinstance(result, dict):
                # 新的查询结果格式
                if 'formula_name' in result and result['formula_name']:
                    formula_name = str(result['formula_name']).strip()

                    # 过滤无效方名与去重检查
                    if (not formula_name) or (formula_name in INVALID_NAMES) or (formula_name in seen_formulas):
                        continue
                    seen_formulas.add(formula_name)

                    # 限制功能主治数量，避免过长
                    functions = result.get('functions', [])
                    if functions and len(functions) > 5:  # 最多显示5个功能主治
                        functions = functions[:5]
                        functions.append("...")

                    # 限制中药组成数量，避免过长
                    herbs = result.get('herbs', [])
                    herb_display = []
                    if herbs and len(herbs) > 8:  # 最多显示8味中药
                        herbs = herbs[:8]
                        herbs.append({"herb": "...", "dose": ""})

                    if herbs:
                        herb_display = [f"{h['herb']}({h['dose']})" for h in herbs if isinstance(h, dict) and 'herb' in h and 'dose' in h]

                    row = {
                        "方剂名称": formula_name,
                        "功能主治": "、".join([f for f in functions if f]) if functions else "未知",
                        "中药组成": "、".join(herb_display) if herb_display else "未知"
                    }
                    table_data.append(row)

                    # 限制表格行数，避免过多数据
                    if len(table_data) >= 20:
                        break

                # 兼容旧的查询结果格式
                else:
                    # 查找药方名称和功能主治
                    formula_name = None
                    herb_function = None

                    for key, value in result.items():
                        if hasattr(value, 'labels') and hasattr(value, 'get'):
                            # Neo4j Node对象
                            if '方名' in value.labels:
                                formula_name = value.get('name', '')
                            elif '功能主治' in value.labels:
                                herb_function = value.get('name', '')
                        elif isinstance(value, dict) and value.get('name'):
                            # 普通字典
                            if key == 'formula_name' or '方名' in str(key):
                                formula_name = value['name']
                            elif key == 'herb_function' or '功能' in str(key):
                                herb_function = value['name']

                    # 如果找到了药方信息，添加到表格中
                    if formula_name:
                        formula_name = str(formula_name).strip()
                        if (not formula_name) or (formula_name in INVALID_NAMES) or (formula_name in seen_formulas):
                            continue
                        seen_formulas.add(formula_name)
                        row = {
                            "方剂名称": formula_name,
                            "功能主治": herb_function if herb_function else "未知",
                            "中药组成": "未知" # 旧格式没有中药组成信息
                        }
                        table_data.append(row)

                        # 限制表格行数
                        if len(table_data) >= 20:
                            break

        return table_data

    def chat(self, query: str, multimodal_inputs: List[Dict] = None, enable_thinking: bool = False) -> Dict:
        """
        主要对话接口 - 实现五步流程

        Args:
            query: 用户问题
            multimodal_inputs: 多模态输入列表

        Returns:
            包含完整流程结果的字典
        """
        start_time = time.time()

        # 处理多模态输入
        if multimodal_inputs:
            combined_text = self.multimodal_processor.combine_multimodal_inputs(multimodal_inputs)
            query = f"{query}\n{combined_text}"

        # 第一步：问题分析和关键词提取
        print("=== 第一步：问题分析和关键词提取 ===")
        # 获取记忆内容，用于问题分析
        memory_content = self.memory.buffer_as_str
        print(f"当前记忆内容: {memory_content}")
        print(f"记忆长度: {len(memory_content) if memory_content else 0}")

        analysis_result = self.query_analyzer.analyze_query(
            query,
            memory_content
        )
        print(f"分析结果: {analysis_result}")

        # 第二步：判断是否需要检索
        need_retrieval = analysis_result["need_retrieval"]
        search_type = analysis_result["search_type"]
        keywords = analysis_result["keywords"]

        if not need_retrieval:
            # 不需要检索，直接生成回复
            print("=== 无需检索，直接生成回复 ===")
            # 传递记忆内容给答案生成器
            answer = self.answer_generator.generate(
                query,
                "",
                "no",
                memory_content=memory_content
            )
            # 保存问题分析结果到记忆中，使用简单的文本格式
            analysis_summary = f"问题: {query} | 分析: 无需检索, 类型: {search_type}, 关键词: {', '.join(keywords)}"
            print(f"保存到记忆: {analysis_summary}")
            self.memory.save_context(
                {"input": query},
                {"output": analysis_summary}
            )
            print(f"保存后记忆内容: {self.memory.buffer_as_str}")

            return {
                "step": "direct_answer",
                "analysis": analysis_result,
                "answer": answer,
                "visualization": None,
                "processing_time": time.time() - start_time
            }

        # 第二步：向量检索
        print("=== 第二步：向量检索 ===")
        # 使用相似度阈值筛选（默认0.6，可根据需要调整/暴露到配置）
        retrieved_nodes = self.vector_retriever.vector_search(keywords, search_type, similarity_threshold=0.6, max_per_keyword=50)
        print(retrieved_nodes)
        print(f"检索到 {len(retrieved_nodes)} 个相关节点")

        # 第三步：构建查询语句
        print("=== 第三步：构建查询语句 ===")
        cypher_queries = self.query_builder.build_queries(retrieved_nodes, search_type)
        print(f"构建了 {len(cypher_queries)} 个查询语句")

        # 第四步：检索结果处理
        print("=== 第四步：检索结果处理 ===")
        # 执行查询
        raw_query_results = []
        if cypher_queries:
            for cypher in cypher_queries:
                try:
                    results = self.graph_db.execute_query(cypher)
                    raw_query_results.extend(results)
                except Exception as e:
                    print(f"查询执行失败: {e}")

        # 先基于原始查询结果统计出现次数（在清洗合并之前进行，避免被去重影响）
        from collections import defaultdict
        INVALID_NAMES = {"未知药方", "未知", "unknown", "Unknown", "UNK"}
        formula_counts: Dict[str, int] = defaultdict(int)
        for raw in raw_query_results:
            try:
                if isinstance(raw, dict):
                    fname = str(raw.get("formula_name", "")).strip()
                    if fname and fname not in INVALID_NAMES:
                        formula_counts[fname] += 1
            except Exception:
                continue

        # 再进行清洗与合并
        cleaned = self._clean_query_results(raw_query_results)
        # 先按出现次数降序排序
        sorted_cleaned = sorted(
            cleaned,
            key=lambda x: formula_counts.get(str(x.get("formula_name", "")).strip(), 0),
            reverse=True
        )
        # 再按阈值过滤（出现次数 >= 阈值 的方剂保留）
        query_results = [
            x for x in sorted_cleaned
            if formula_counts.get(str(x.get("formula_name", "")).strip(), 0) >= 0.7 * len(keywords)
        ]
        # 可选：限制前端规模（例如最多50个方剂），如需完整可视化可去掉此限制
        # query_results = query_results[:50]

        # 药方信息总结（仅名称+功能主治）
        print("=== 药方信息总结 ===")
        formula_summary = self._summarize_formulas(query_results[:20])

        # 生成表格数据用于前端展示（已在内部再次排除无效方名）
        table_data = self._generate_table_data(query_results)
        # 生成可视化结果
        visualization = self.result_visualizer.visualize_results(query_results)

        # 格式化结果用于生成回复（使用总结后的药方信息）
        formatted_results = formula_summary

        # 第五步：生成最终回复
        print("=== 第五步：生成回复 ===")
        print(f"传递给答案生成器的记忆内容: {memory_content}")
        # 统一限制输出长度512；按是否思考模式设置采样参数
        if enable_thinking:
            answer = self.answer_generator.generate(
                query,
                formatted_results,
                "no",
                enable_thinking=True,
                memory_content=memory_content,
                max_new_tokens=1024,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
            )
        else:
            answer = self.answer_generator.generate(
                query,
                formatted_results,
                "no",
                enable_thinking=False,
                memory_content=memory_content,
                max_new_tokens=768,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
            )

        # 保存对话记忆 - 保存问题分析结果，使用简单的文本格式
        analysis_summary = f"问题: {query} | 分析: 需要检索, 类型: {search_type}, 关键词: {', '.join(keywords)}"
        print(f"保存到记忆: {analysis_summary}")
        self.memory.save_context(
            {"input": query},
            {"output": analysis_summary}
        )
        print(f"保存后记忆内容: {self.memory.buffer_as_str}")

        end_time = time.time()

        return {
            "step": "retrieval_based",
            "analysis": analysis_result,
            "retrieved_nodes": retrieved_nodes,
            "cypher_queries": cypher_queries,
            "query_results": query_results,  # 返回清洗合并后的结果
            "formula_summary": formula_summary,
            "table_data": table_data,
            "visualization": visualization,
            "answer": answer,
            "processing_time": end_time - start_time
        }

    def _format_results_for_answer(self, query_results: List[Dict]) -> str:
        """格式化查询结果用于生成回复"""
        if not query_results:
            return "未找到相关信息"

        formatted = []
        for result in query_results:
            if isinstance(result, dict):
                # 新的查询结果格式
                if 'formula_name' in result and result['formula_name']:
                    formula_name = result['formula_name']
                    formatted.append(f"方剂名称: {formula_name}")

                    # 功能主治
                    if 'functions' in result and result['functions']:
                        functions_str = "、".join([f for f in result['functions'] if f])
                        if functions_str:
                            formatted.append(f"功能主治: {functions_str}")

                    # 中药组成
                    if 'herbs' in result and result['herbs']:
                        herbs_str = "、".join([f"{h['herb']}({h['dose']})" for h in result['herbs'] if isinstance(h, dict) and 'herb' in h and 'dose' in h])
                        if herbs_str:
                            formatted.append(f"中药组成: {herbs_str}")

                    # 来源和别名
                    if 'sources' in result and result['sources']:
                        sources_str = "、".join([s for s in result['sources'] if s])
                        if sources_str:
                            formatted.append(f"来源: {sources_str}")

                    if 'aliases' in result and result['aliases']:
                        aliases_str = "、".join([a for a in result['aliases'] if a])
                        if aliases_str:
                            formatted.append(f"别名: {aliases_str}")

                # 兼容旧的查询结果格式
                else:
                    for key, value in result.items():
                        # 检查是否是Neo4j Node对象
                        if hasattr(value, 'labels') and hasattr(value, 'get'):
                            # 这是Neo4j Node对象
                            node_name = value.get('name', '')
                            if node_name:
                                node_type = list(value.labels)[0] if value.labels else key
                                formatted.append(f"{node_type}: {node_name}")
                        elif isinstance(value, dict) and value.get('name'):
                            # 这是普通字典
                            formatted.append(f"{key}: {value['name']}")
                        elif isinstance(value, str) and value:
                            # 这是字符串值（如herb_weight）
                            formatted.append(f"{key}: {value}")

        return "\n".join(formatted) if formatted else "找到相关信息但格式不标准"

    def get_memory(self) -> str:
        """获取对话记忆"""
        return self.memory.buffer_as_str

    def get_memory_debug(self) -> Dict:
        """获取记忆调试信息"""
        try:
            memory_buffer = self.memory.buffer_as_str
            memory_variables = self.memory.memory_variables
            memory_key = self.memory.memory_key
            return {
                "buffer_as_str": memory_buffer,
                "memory_variables": memory_variables,
                "memory_key": memory_key,
                "buffer_length": len(memory_buffer) if memory_buffer else 0
            }
        except Exception as e:
            return {"error": str(e)}

    def refresh_memory(self, k: int = 20):
        """刷新对话记忆"""
        self.memory = ConversationBufferWindowMemory(k=k)


class AnswerGenerator:
    """答案生成器"""

    def __init__(self, api_key: str = None, memory_size: int = 20):
        self.local_model = get_local_model()
        # 不再维护独立的记忆系统，使用主引擎传递的记忆内容

    def generate(self, query: str, kg_information: str = "", stream: str = 'no', enable_thinking: bool = False, memory_content: str = "", **kwargs) -> str:
        """
        生成答案

        Args:
            query: 用户问题
            kg_information: 知识图谱信息
            stream: 是否流式输出
            memory_content: 对话记忆内容

        Returns:
            生成的答案
        """
        # 2. 如果涉及具体方剂，要详细说明组成、功效、用法
        # 3. 如果涉及症状，要给出合理的治疗建议
        prompt = f"""
        你是一个专业的中医问答助手。请根据用户问题、知识图谱信息和对话历史生成专业、准确的回答。

        对话历史：
        {memory_content if memory_content else "无对话历史"}
        
        用户问题：{query}
        
        知识图谱信息：
        {kg_information if kg_information else "无相关信息"}
        
        请生成专业的中医回答，要求：
        1. 回答要准确、专业
        2. 语言要通俗易懂
        3. 如果不确定，要明确说明
        4. 根据药方的功能主治，给出患者可能需要补充的症状建议
        5. 注意字数不要太多，信息紧凑
        6. 如果对话历史中有相关信息，请适当参考并保持回答的连贯性
        """
        try:
            response = self.local_model.call(
                messages=[
                    {"role": "system", "content": "你是一个专业的中医问答助手"},
                    {"role": "user", "content": prompt}
                ],
                enable_thinking=enable_thinking,  # 根据参数控制思考功能
                **kwargs
            )
            # print(f"DEBUG: local_model.call返回结果: {response}")

            if response["status_code"] == 200:
                answer = response["output"]["choices"][0]["message"]["content"]
                return answer
            else:
                return "抱歉，生成答案时出现错误，请稍后重试。"

        except Exception as e:
            print(f"生成答案失败: {e}")
            import traceback
            traceback.print_exc()
            return "抱歉，生成答案时出现错误，请稍后重试。"

    def get_memory(self) -> str:
        """获取对话记忆 - 已废弃，使用主引擎的记忆系统"""
        return "记忆系统已集成到主引擎中"

    def save_memory(self, query: str, answer: str):
        """保存对话记忆 - 已废弃，使用主引擎的记忆系统"""
        pass

    def refresh_memory(self, k: int = 20):
        """刷新对话记忆 - 已废弃，使用主引擎的记忆系统"""
        pass