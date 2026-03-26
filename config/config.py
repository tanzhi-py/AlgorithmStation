import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """配置类，整合所有配置参数"""
    
    # Neo4j数据库配置
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '1573028PengYao@')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    # 千问API配置
    QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'sk-a5f284f067f742c38ffe48c0eb11d854')
    QWEN_BASE_URL = os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/api/v1')
    QWEN_MODEL = os.getenv('QWEN_MODEL', 'qwen-max')
    
    # OpenAI配置（备用）
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # 应用配置
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'false'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # 文件上传配置
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # 知识图谱配置
    KNOWLEDGE_GRAPH_FILE = os.getenv('KNOWLEDGE_GRAPH_FILE', 'tcm_knowledge_graph.json')
    CRAWLER_DATA_FOLDER = os.getenv('CRAWLER_DATA_FOLDER', 'crawled_data')
    
    # RAG配置
    MAX_QUERY_RESULTS = int(os.getenv('MAX_QUERY_RESULTS', 10))
    MEMORY_SIZE = int(os.getenv('MEMORY_SIZE', 5))
    CHUNK_LENGTH = int(os.getenv('CHUNK_LENGTH', 1000))
    TOKEN_LIMIT = int(os.getenv('TOKEN_LIMIT', 4000))
    
    # 性能优化配置
    FAST_MODE_ENABLED = os.getenv('FAST_MODE_ENABLED', 'True').lower() == 'true'
    THINKING_MODE_ENABLED = os.getenv('THINKING_MODE_ENABLED', 'False').lower() == 'true'
    VISUALIZATION_ENABLED = os.getenv('VISUALIZATION_ENABLED', 'True').lower() == 'true'
    DEBUG_OUTPUT_ENABLED = os.getenv('DEBUG_OUTPUT_ENABLED', 'False').lower() == 'false'
    MAX_PROCESSING_TIME = int(os.getenv('MAX_PROCESSING_TIME', 30))  # 最大处理时间（秒）
    
    # 多模态配置
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
    ALLOWED_TEXT_EXTENSIONS = {'txt', 'md', 'json'}
    
    # 图谱可视化配置
    GRAPH_NODE_LIMIT = int(os.getenv('GRAPH_NODE_LIMIT', 100))
    GRAPH_RELATION_LIMIT = int(os.getenv('GRAPH_RELATION_LIMIT', 200))
    
    # 缓存配置
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 3600))  # 1小时
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    
    @classmethod
    def validate_config(cls):
        """验证配置参数"""
        required_configs = [
            'QWEN_API_KEY',
            'NEO4J_URI',
            'NEO4J_USER',
            'NEO4J_PASSWORD'
        ]
        
        missing_configs = []
        for config_name in required_configs:
            if not getattr(cls, config_name):
                missing_configs.append(config_name)
        
        if missing_configs:
            raise ValueError(f"缺少必要的配置参数: {', '.join(missing_configs)}")
        
        return True
    
    @classmethod
    def get_database_url(cls):
        """获取数据库连接URL"""
        return f"{cls.NEO4J_URI}"
    
    @classmethod
    def get_upload_path(cls):
        """获取上传文件路径"""
        return os.path.join(os.getcwd(), cls.UPLOAD_FOLDER)
    
    @classmethod
    def get_knowledge_graph_path(cls):
        """获取知识图谱文件路径"""
        return os.path.join(os.getcwd(), cls.KNOWLEDGE_GRAPH_FILE)
    
    @classmethod
    def get_crawler_data_path(cls):
        """获取爬虫数据路径"""
        return os.path.join(os.getcwd(), cls.CRAWLER_DATA_FOLDER) 