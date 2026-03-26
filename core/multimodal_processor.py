import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from typing import Dict, List, Optional, Tuple
import json
import re
import threading
import base64
from pydub import AudioSegment
import tempfile
import os

# 尝试导入speech_recognition，如果失败则使用模拟版本
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("警告: speech_recognition 模块未安装，音频处理功能将被禁用")

# 尝试导入dashscope，如果失败则使用模拟版本
try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告: dashscope 模块未安装，多模态API功能将被禁用")

edited = True
class MultimodalProcessor:
    """多模态处理器，支持文本、图片、音频输入"""
    
    def __init__(self, api_key: str):
        """
        初始化多模态处理器
        
        Args:
            api_key: 千问API密钥
        """
        self.api_key = api_key
        if DASHSCOPE_AVAILABLE:
            dashscope.api_key = api_key
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
        else:
            self.recognizer = None
        
    def process_text(self, text: str) -> Dict:
        """
        处理文本输入
        
        Args:
            text: 输入文本
            
        Returns:
            处理结果
        """
        return {
            "type": "text",
            "content": text,
            "processed": True
        }

    def process_image(self, image_path: str) -> Dict:
        """
        处理图片输入，提取图片中的文本信息（使用 threading 实现超时，兼容 Windows）
        将本地图片转为 Base64 后调用 dashscope API
        """
        try:
            # 检查图片是否可读
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "type": "image",
                    "content": "无法读取图片，请检查文件格式或路径",
                    "processed": True,
                    "error": "无法读取图片",
                    "original_path": image_path
                }

            # 检查 dashscope 是否可用
            if not DASHSCOPE_AVAILABLE:
                return {
                    "type": "image",
                    "content": "图片已上传，但 dashscope 模块未安装，无法进行详细分析",
                    "processed": True,
                    "original_path": image_path
                }

            # 将图片转换为 Base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            # 根据图片类型设置 MIME 类型
            ext = os.path.splitext(image_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                mime = 'image/jpeg'
            elif ext == '.png':
                mime = 'image/png'
            elif ext == '.gif':
                mime = 'image/gif'
            else:
                mime = 'image/jpeg'
            image_data_uri = f"data:{mime};base64,{image_base64}"

            # 定义超时时间（秒）
            TIMEOUT = 30
            result_container = []  # 存放成功响应
            error_container = []  # 存放异常

            def target():
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"image": image_data_uri},
                                {
                                    "text": "请分析这张图片中的内容，如果是中药相关的图片，请提取其中的中药名称、症状描述等信息。如果是手写文字，请识别其中的文字内容。"}
                            ]
                        }
                    ]
                    # 有些模型版本可能要求使用 "image_url" 字段，请根据文档调整
                    response = MultiModalConversation.call(
                        model="qwen-vl-plus",  # 或 "qwen-vl-max"
                        messages=messages
                    )
                    result_container.append(response)
                except Exception as e:
                    error_container.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(TIMEOUT)

            # 超时处理
            if thread.is_alive():
                return {
                    "type": "image",
                    "content": "图片分析超时，请稍后重试",
                    "processed": True,
                    "original_path": image_path
                }

            if error_container:
                raise error_container[0]

            if not result_container:
                return {
                    "type": "image",
                    "content": "图片分析未返回结果",
                    "processed": True,
                    "original_path": image_path
                }

            response = result_container[0]
            if response.status_code == 200:
                # 根据实际返回结构提取文本
                extracted_text = response.output.choices[0].message.content
                return {
                    "type": "image",
                    "content": extracted_text,
                    "processed": True,
                    "original_path": image_path
                }
            else:
                return {
                    "type": "image",
                    "content": f"图片分析 API 调用失败: {response.message} (code: {response.status_code})",
                    "processed": True,
                    "error": response.message,
                    "original_path": image_path
                }

        except Exception as e:
            return {
                "type": "image",
                "content": f"图片分析失败：{str(e)}",
                "processed": True,
                "error": str(e),
                "original_path": image_path
            }
    
    def process_audio_before(self, audio_path: str) -> Dict:
        """
        处理音频输入，转换为文本
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            处理结果
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            return {
                "type": "audio",
                "content": "",
                "error": "speech_recognition 模块未安装，无法处理音频"
            }
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("音频识别超时")
            
            # 设置30秒超时
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                
            # 使用Google Speech Recognition
            text = self.recognizer.recognize_google(audio, language='zh-CN')
            
            # 取消超时
            signal.alarm(0)
            
            return {
                "type": "audio",
                "content": text,
                "processed": True,
                "original_path": audio_path
            }
            
        except sr.UnknownValueError:
            return {
                "type": "audio",
                "content": "",
                "error": "无法识别音频内容"
            }
        except sr.RequestError as e:
            return {
                "type": "audio",
                "content": "",
                "error": f"语音识别服务出错: {str(e)}"
            }
        except TimeoutError:
            return {
                "type": "audio",
                "content": "音频识别超时，请稍后重试",
                "processed": True,
                "original_path": audio_path
            }
        except Exception as e:
            return {
                "type": "audio",
                "content": "",
                "error": f"处理音频时出错: {str(e)}"
            }

    def process_audio(self, audio_path: str) -> Dict:
        print(f"开始处理音频: {audio_path}")
        if not os.path.exists(audio_path):
            return {"type": "audio", "content": "", "error": "文件不存在", "original_path": audio_path}

        # 如果需要格式转换（Whisper 支持多种格式，可以省略）
        wav_path = audio_path
        cleanup_needed = False
        if not audio_path.lower().endswith('.wav'):
            try:
                audio = AudioSegment.from_file(audio_path)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    wav_path = tmp.name
                audio.export(wav_path, format='wav')
                cleanup_needed = True
            except Exception as e:
                return {"type": "audio", "content": "", "error": f"转换失败: {str(e)}", "original_path": audio_path}

        TIMEOUT = 30
        result_container = []
        error_container = []

        def target():
            try:
                # 使用 Whisper 识别
                if not hasattr(self, 'whisper_model') or self.whisper_model is None:
                    import whisper
                    self.whisper_model = whisper.load_model("base")
                text = self.whisper_model.transcribe(wav_path, language='zh')['text']
                result_container.append(text)
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_container.append(str(e))

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(TIMEOUT)

        if cleanup_needed and os.path.exists(wav_path):
            os.unlink(wav_path)

        if thread.is_alive():
            return {"type": "audio", "content": "", "error": "识别超时", "original_path": audio_path}
        if error_container:
            return {"type": "audio", "content": "", "error": error_container[0], "original_path": audio_path}
        return {"type": "audio", "content": result_container[0], "processed": True, "original_path": audio_path}
    
    def extract_medical_entities(self, text: str) -> Dict:
        """
        从文本中提取医学实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的医学实体
        """
        try:
            prompt = f"""
            请从以下文本中提取中医相关的实体信息，包括：
            1. 症状描述
            2. 中药名称
            3. 方剂名称
            4. 疾病名称
            5. 治疗建议
            
            文本内容：{text}
            
            请以JSON格式返回，包含以下字段：
            - symptoms: 症状列表
            - herbs: 中药名称列表
            - formulas: 方剂名称列表
            - diseases: 疾病名称列表
            - suggestions: 治疗建议
            """
            
            response = dashscope.Generation.call(
                model="qwen-max",
                messages=[
                    {"role": "system", "content": "你是一个专业的中医实体识别助手"},
                    {"role": "user", "content": prompt}
                ],
                result_format='message'
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                try:
                    # 尝试解析JSON
                    entities = json.loads(content)
                    return {
                        "success": True,
                        "entities": entities
                    }
                except json.JSONDecodeError:
                    # 如果JSON解析失败，使用正则表达式提取
                    return self._extract_entities_with_regex(text)
            else:
                return self._extract_entities_with_regex(text)
                
        except Exception as e:
            return self._extract_entities_with_regex(text)
    
    def _extract_entities_with_regex(self, text: str) -> Dict:
        """
        使用正则表达式提取实体（备用方案）
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体
        """
        # 常见中药名称模式
        herb_patterns = [
            r'[人参|黄芪|当归|白芍|川芎|茯苓|白术|甘草|陈皮|半夏|柴胡|黄芩|黄连|黄柏|栀子|连翘|金银花|板蓝根|大青叶|蒲公英|紫花地丁|鱼腥草|败酱草|红藤|白花蛇舌草|半边莲|半枝莲|白毛夏枯草|白花蛇舌草|半边莲|半枝莲|白毛夏枯草|白花蛇舌草|半边莲|半枝莲|白毛夏枯草]{2,4}',
            r'[根|茎|叶|花|果|皮|仁|子|草|藤|木|香|砂|仁|子|草|藤|木|香|砂]{1,2}[根|茎|叶|花|果|皮|仁|子|草|藤|木|香|砂]{1,2}'
        ]
        
        # 常见症状模式
        symptom_patterns = [
            r'[头痛|头晕|恶心|呕吐|腹痛|腹泻|便秘|咳嗽|咳痰|气喘|胸闷|心悸|失眠|多梦|烦躁|易怒|抑郁|焦虑|疲劳|乏力|食欲不振|口干|口苦|口臭|口淡|口甜|口酸|口咸|口辣|口麻|口涩|口腻|口粘|口干|口苦|口臭|口淡|口甜|口酸|口咸|口辣|口麻|口涩|口腻|口粘]{2,4}',
            r'[痛|酸|麻|胀|闷|重|轻|冷|热|寒|温|凉|燥|湿|虚|实|表|里|阴|阳]{1,2}[痛|酸|麻|胀|闷|重|轻|冷|热|寒|温|凉|燥|湿|虚|实|表|里|阴|阳]{1,2}'
        ]
        
        herbs = []
        symptoms = []
        
        # 提取中药名称
        for pattern in herb_patterns:
            matches = re.findall(pattern, text)
            herbs.extend(matches)
        
        # 提取症状
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text)
            symptoms.extend(matches)
        
        return {
            "success": True,
            "entities": {
                "symptoms": list(set(symptoms)),
                "herbs": list(set(herbs)),
                "formulas": [],
                "diseases": [],
                "suggestions": []
            }
        }
    
    def combine_multimodal_inputs(self, inputs: List[Dict]) -> str:
        """
        合并多模态输入
        
        Args:
            inputs: 多模态输入列表
            
        Returns:
            合并后的文本
        """
        combined_text = ""
        
        for input_data in inputs:
            if input_data.get("processed") and input_data.get("content"):
                if input_data["type"] == "text":
                    combined_text += f"文本输入：{input_data['content']}\n"
                elif input_data["type"] == "image":
                    combined_text += f"图片内容：{input_data['content']}\n"
                elif input_data["type"] == "audio":
                    combined_text += f"语音内容：{input_data['content']}\n"
        
        return combined_text.strip()
    
    def validate_input(self, input_data: Dict) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            是否有效
        """
        if not input_data:
            return False
        
        if input_data.get("type") not in ["text", "image", "audio"]:
            return False
        
        if input_data.get("type") == "text" and not input_data.get("content"):
            return False
        
        if input_data.get("type") in ["image", "audio"] and not input_data.get("original_path"):
            return False
        
        return True 