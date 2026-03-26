import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Dict, Optional, Any
import os

class LocalQwenModel:
    """本地 Qwen3-8B 模型调用类"""

    def __init__(self, model_path: str = r"../kg_llm/models/qwen"):
        """
        初始化本地模型

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """加载模型和分词器"""
        try:
            print(f"正在加载模型: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            print("模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        min_p: float = 0.0,  # 预留参数，HF默认不使用
        enable_thinking: bool = False,
    ) -> str:
        """
        生成回复

        Args:
            messages: 对话消息列表
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            enable_thinking: 是否启用思考功能

        Returns:
            生成的回复
        """
        try:
            # 构建对话模板
            if len(messages) == 1:
                # 单条消息
                prompt = messages[0]["content"]
            else:
                # 多条消息，构建对话格式
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking  # 在tokenizer中设置思考模式
                )

            # 编码输入
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

            # 生成回复
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 解码输出
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response.strip()

        except Exception as e:
            print(f"生成回复失败: {e}")
            return "抱歉，生成回复时出现错误。"

    def call(self, model: str = None, messages: List[Dict] = None, result_format: str = 'message', enable_thinking: bool = False, **kwargs) -> Dict:
        """
        模拟 dashscope.Generation.call 接口

        Args:
            model: 模型名称（忽略）
            messages: 对话消息列表
            result_format: 返回格式
            enable_thinking: 是否启用思考功能
            **kwargs: 其他参数

        Returns:
            模拟的 API 响应
        """
        if not messages:
            return {
                "status_code": 400,
                "code": "INVALID_INPUT",
                "message": "messages parameter is required"
            }

        try:
            response_content = self.generate(messages, enable_thinking=enable_thinking, **kwargs)

            return {
                "status_code": 200,
                "code": "SUCCESS",
                "message": "Success",
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": response_content
                            }
                        }
                    ]
                }
            }

        except Exception as e:
            return {
                "status_code": 500,
                "code": "GENERATION_ERROR",
                "message": str(e)
            }


# 全局模型实例
_local_model = None

def get_local_model() -> LocalQwenModel:
    """获取全局模型实例"""
    global _local_model
    if _local_model is None:
        _local_model = LocalQwenModel()
    return _local_model