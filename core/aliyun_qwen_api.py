#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云千问API调用模块
"""

import dashscope
from dashscope import Generation
from typing import List, Dict, Optional, Any
import json


class AliyunQwenAPI:
    """阿里云千问API调用类"""

    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-max"):
        """
        初始化阿里云千问API

        Args:
            api_key: API密钥
            base_url: API基础URL（可为None，使用默认）
            model: 模型名称
        """
        self.api_key = api_key
        self.model = model

        # 设置API密钥
        dashscope.api_key = api_key

        # 如果有自定义base_url，设置它
        if base_url:
            dashscope.base_http_api_url = base_url

    def call(
            self,
            messages: List[Dict],
            model: str = None,
            result_format: str = 'message',
            enable_thinking: bool = False,
            **kwargs
    ) -> Dict:
        """
        调用千问API

        Args:
            messages: 对话消息列表
            model: 模型名称（如不指定则使用初始化时的模型）
            result_format: 返回格式
            enable_thinking: 是否启用思考功能
            **kwargs: 其他参数

        Returns:
            API响应结果
        """
        try:
            # 准备参数
            model_name = model or self.model

            # 构建调用参数
            call_kwargs = {
                "model": model_name,
                "messages": messages,
                "result_format": result_format
            }

            # 添加思考模式参数（关键修复）
            if enable_thinking:
                call_kwargs['enable_thinking'] = True

            # 添加可选参数
            if 'temperature' in kwargs:
                call_kwargs['temperature'] = kwargs['temperature']
            if 'top_p' in kwargs:
                call_kwargs['top_p'] = kwargs['top_p']
            if 'top_k' in kwargs:
                call_kwargs['top_k'] = kwargs['top_k']
            if 'max_tokens' in kwargs:
                call_kwargs['max_tokens'] = kwargs['max_tokens']
            elif 'max_new_tokens' in kwargs:
                call_kwargs['max_tokens'] = kwargs['max_new_tokens']

            # 调用API
            response = Generation.call(**call_kwargs)

            # 统一返回格式
            if response.status_code == 200:
                return {
                    "status_code": 200,
                    "code": "SUCCESS",
                    "message": "Success",
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": response.output.choices[0].message.content
                                }
                            }
                        ]
                    }
                }
            else:
                return {
                    "status_code": response.status_code,
                    "code": response.code,
                    "message": response.message
                }

        except Exception as e:
            return {
                "status_code": 500,
                "code": "API_ERROR",
                "message": f"API调用异常: {str(e)}"
            }

    def chat(
            self,
            prompt: str,
            system_prompt: str = "你是一个专业的中医问答助手",
            **kwargs
    ) -> str:
        """
        简化的聊天接口

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数

        Returns:
            模型回复
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.call(messages, **kwargs)

        if response["status_code"] == 200:
            return response["output"]["choices"][0]["message"]["content"]
        else:
            return f"API调用失败: {response['message']}"

    def stream_chat(self, messages: List[Dict], **kwargs):
        """
        流式聊天接口

        Args:
            messages: 对话消息列表
            **kwargs: 其他参数

        Returns:
            生成器，返回流式响应
        """
        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                result_format='message',
                stream=True,
                **kwargs
            )

            for chunk in response:
                if chunk.status_code == 200:
                    if hasattr(chunk.output, 'choices') and chunk.output.choices:
                        content = chunk.output.choices[0].message.content
                        if content:
                            yield content
        except Exception as e:
            yield f"流式调用异常: {str(e)}"


# 全局API实例
_aliyun_api = None


def get_local_model() -> AliyunQwenAPI:
    """获取全局阿里云API实例"""
    global _aliyun_api
    if _aliyun_api is None:
        from config.config import Config
        config = Config()
        _aliyun_api = AliyunQwenAPI(
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL,
            model=config.QWEN_MODEL
        )
    return _aliyun_api


def test_aliyun_api():
    """测试阿里云API连接"""
    try:
        api = get_local_model()

        test_messages = [
            {"role": "system", "content": "你是一个测试助手"},
            {"role": "user", "content": "你好，请简单介绍一下自己"}
        ]

        response = api.call(test_messages)

        if response["status_code"] == 200:
            print("✓ 阿里云千问API连接成功")
            print(f"模型回复: {response['output']['choices'][0]['message']['content'][:100]}...")
            return True
        else:
            print(f"✗ 阿里云千问API连接失败: {response['message']}")
            return False

    except Exception as e:
        print(f"✗ 阿里云千问API连接异常: {e}")
        return False


if __name__ == "__main__":
    # 测试API
    test_aliyun_api()