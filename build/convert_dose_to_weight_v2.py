#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将现有JSON文件中的dose关系转换为composition关系的权重属性
按照方剂分组处理，确保中药组成和剂量一一对应
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_formula_structure(triples: List[Dict]) -> Dict[str, Dict]:
    """
    解析三元组，按照方剂分组构建结构
    
    Args:
        triples: 三元组列表
        
    Returns:
        按方剂分组的字典结构
    """
    formula_structure = defaultdict(lambda: {
        'formula_name': '',
        'prescription': '',
        'herbs': [],
        'functions': [],
        'sources': [],
        'aliases': []
    })
    
    current_formula = None
    
    for triple in triples:
        relation = triple["relation"]
        
        if relation == "include":
            # 新的方剂开始
            formula_name = triple["node_2"].split("\t")[1]
            current_formula = formula_name
            formula_structure[formula_name]['formula_name'] = formula_name
            
        elif relation == "prescription type" and current_formula:
            # 处方信息
            prescription_name = triple["node_2"].split("\t")[1]
            formula_structure[current_formula]['prescription'] = prescription_name
            
        elif relation == "composition" and current_formula:
            # 中药组成
            herb_name = triple["node_2"].split("\t")[1]
            formula_structure[current_formula]['herbs'].append({
                'name': herb_name,
                'dose': None
            })
            
        elif relation == "dose" and current_formula:
            # 剂量信息，需要匹配到对应的中药
            herb_name = triple["node_1"].split("\t")[1]
            dose_value = triple["node_2"].split("\t")[1]
            
            # 查找对应的中药并添加剂量
            for herb in formula_structure[current_formula]['herbs']:
                if herb['name'] == herb_name:
                    herb['dose'] = dose_value
                    break
                    
        elif relation == "functions" and current_formula:
            # 功能主治
            function = triple["node_2"].split("\t")[1]
            formula_structure[current_formula]['functions'].append(function)
            
        elif relation == "from" and current_formula:
            # 来源
            source = triple["node_2"].split("\t")[1]
            formula_structure[current_formula]['sources'].append(source)
            
        elif relation == "another name" and current_formula:
            # 别名
            alias = triple["node_2"].split("\t")[1]
            formula_structure[current_formula]['aliases'].append(alias)
    
    return formula_structure

def convert_to_weighted_triples(formula_structure: Dict) -> List[Dict]:
    """
    将方剂结构转换为带权重的三元组
    
    Args:
        formula_structure: 方剂结构字典
        
    Returns:
        新的三元组列表
    """
    new_triples = []
    
    for formula_name, structure in formula_structure.items():
        # 1. 方剂包含方名
        new_triples.append({
            "node_1": "方剂\t方剂",
            "relation": "include",
            "node_2": f"方名\t{formula_name}"
        })
        
        # 2. 方名到处方的关系
        if structure['prescription']:
            new_triples.append({
                "node_1": f"方名\t{formula_name}",
                "relation": "prescription type",
                "node_2": f"处方\t{structure['prescription']}"
            })
        
        # 3. 处方到中药的组成关系（带权重）
        for herb in structure['herbs']:
            if herb['dose']:
                # 有剂量信息，添加权重
                new_triples.append({
                    "node_1": f"处方\t{structure['prescription']}",
                    "relation": "composition",
                    "node_2": f"中药名\t{herb['name']}",
                    "weight": herb['dose']
                })
            else:
                # 没有剂量信息，不添加权重
                new_triples.append({
                    "node_1": f"处方\t{structure['prescription']}",
                    "relation": "composition",
                    "node_2": f"中药名\t{herb['name']}"
                })
        
        # 4. 功能主治
        for function in structure['functions']:
            new_triples.append({
                "node_1": f"方名\t{formula_name}",
                "relation": "functions",
                "node_2": f"功能主治\t{function}"
            })
        
        # 5. 来源
        for source in structure['sources']:
            new_triples.append({
                "node_1": f"方名\t{formula_name}",
                "relation": "from",
                "node_2": f"来源\t{source}"
            })
        
        # 6. 别名
        for alias in structure['aliases']:
            new_triples.append({
                "node_1": f"方名\t{formula_name}",
                "relation": "another name",
                "node_2": f"别名\t{alias}"
            })
    
    return new_triples

def convert_file_with_formula_grouping(input_file: str, output_file: str):
    """
    使用方剂分组方法转换文件
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return
    
    print(f"开始处理文件: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        triples = json.load(f)
    
    print(f"原始三元组数量: {len(triples)}")
    
    # 按方剂分组解析结构
    print("正在按方剂分组解析...")
    formula_structure = parse_formula_structure(triples)
    
    print(f"识别到 {len(formula_structure)} 个方剂")
    
    # 显示一些方剂的结构示例
    for i, (formula_name, structure) in enumerate(list(formula_structure.items())[:3]):
        print(f"\n方剂 {i+1}: {formula_name}")
        print(f"  中药数量: {len(structure['herbs'])}")
        print(f"  有剂量的中药: {sum(1 for h in structure['herbs'] if h['dose'])}")
        print(f"  功能主治数量: {len(structure['functions'])}")
    
    # 转换为带权重的三元组
    print("\n正在转换为带权重的三元组...")
    new_triples = convert_to_weighted_triples(formula_structure)
    
    print(f"转换后三元组数量: {len(new_triples)}")
    
    # 统计权重信息
    weight_count = sum(1 for triple in new_triples if "weight" in triple)
    print(f"包含权重的三元组数量: {weight_count}")
    
    # 保存到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_triples, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成，结果已保存到: {output_file}")
    
    return new_triples

def convert_all_backup_files():
    """转换所有备份文件"""
    backup_files = [
        "../kg_llm/pdf/pdf_knowledge_graph_backup.json",
        "../kg_llm/webinff/tcm_knowledge_graph_backup.json"
    ]
    
    for backup_file in backup_files:
        if os.path.exists(backup_file):
            print(f"\n{'='*60}")
            print(f"处理备份文件: {backup_file}")
            print(f"{'='*60}")
            
            # 生成新文件名
            base_name = os.path.basename(backup_file)
            name_without_backup = base_name.replace('_backup.json', '')
            output_file = f"../kg_llm/{name_without_backup}_weighted.json"
            
            # 转换文件
            convert_file_with_formula_grouping(backup_file, output_file)
        else:
            print(f"备份文件不存在，跳过: {backup_file}")

def verify_conversion_results():
    """验证转换结果"""
    print(f"\n{'='*60}")
    print("验证转换结果")
    print(f"{'='*60}")
    
    result_files = [
        "../kg_llm/pdf/pdf_knowledge_graph_weighted.json",
        "../kg_llm/webinff/tcm_knowledge_graph_weighted.json"
    ]
    
    for result_file in result_files:
        if os.path.exists(result_file):
            print(f"\n验证文件: {result_file}")
            
            with open(result_file, 'r', encoding='utf-8') as f:
                triples = json.load(f)
            
            # 统计信息
            total_triples = len(triples)
            dose_relations = sum(1 for t in triples if t["relation"] == "dose")
            composition_with_weight = sum(1 for t in triples if t["relation"] == "composition" and "weight" in t)
            composition_without_weight = sum(1 for t in triples if t["relation"] == "composition" and "weight" not in t)
            
            print(f"  总三元组数: {total_triples}")
            print(f"  dose关系数: {dose_relations}")
            print(f"  带权重的composition关系数: {composition_with_weight}")
            print(f"  不带权重的composition关系数: {composition_without_weight}")
            
            # 显示一些带权重的示例
            weight_examples = [t for t in triples if t["relation"] == "composition" and "weight" in t][:3]
            if weight_examples:
                print(f"  带权重的composition关系示例:")
                for i, example in enumerate(weight_examples, 1):
                    herb_name = example["node_2"].split("\t")[1]
                    weight = example["weight"]
                    print(f"    {i}. {herb_name}: {weight}")

if __name__ == "__main__":
    print("开始使用方剂分组方法转换dose关系到权重属性...")
    
    # 转换所有备份文件
    convert_all_backup_files()
    
    # 验证转换结果
    verify_conversion_results()
    
    print("\n转换完成！所有结果已保存到新文件中。") 