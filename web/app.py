import os
import json
import sys
from datetime import datetime, timezone, time
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_socketio import SocketIO, emit, join_room, leave_room

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_database import GraphDatabase
from core.optimized_rag_engine import OptimizedRAGEngine
from core.multimodal_processor import MultimodalProcessor
from config.config import Config
import eventlet
eventlet.monkey_patch()

# ================= 应用与扩展初始化 ==================
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tcm.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 初始化扩展
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins="*")

# ================ 数据库模型定义 ===================
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    doctor_messages = db.relationship('DoctorMessage', backref='user', lazy=True)

class Conversation(db.Model):
    __tablename__ = 'conversations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    messages = db.relationship('Message', backref='conversation', lazy=True, order_by='Message.timestamp')

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    role = db.Column(db.String(10))
    content = db.Column(db.Text)
    multimodal_data = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class Doctor(db.Model):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    specialty = db.Column(db.String(200))
    work_start = db.Column(db.Time, nullable=False)
    work_end = db.Column(db.Time, nullable=False)
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)

class DoctorMessage(db.Model):
    __tablename__ = 'doctor_messages'
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    sender_role = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    doctor = db.relationship('Doctor', backref='messages')

# ================ 核心组件初始化 ==================
config = Config()
graph_db = GraphDatabase(
    uri=config.NEO4J_URI,
    user=config.NEO4J_USER,
    password=config.NEO4J_PASSWORD,
    database=config.NEO4J_DATABASE
)
rag_engine = OptimizedRAGEngine(graph_db, config.QWEN_API_KEY)
multimodal_processor = MultimodalProcessor(config.QWEN_API_KEY)

# ================ 辅助函数 ==================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_auto_reply(user_message, doctor_id):
    doctor = Doctor.query.get(doctor_id)
    if doctor:
        return f"【{doctor.name} 医生】您好，根据您的描述，我为您分析如下：\n\n（系统自动回复）{user_message[:100]}...\n\n建议您提供更多细节。"
    else:
        return "系统暂时无法回复，请稍后再试。"

# =============== 用户认证路由 =================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            error = '用户名已存在'
        else:
            user = User(
                username=username,
                password_hash=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for('detailed_chat'))
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = True if request.form.get('remember') else False
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            return redirect(url_for('detailed_chat'))
        else:
            error = '用户名或密码错误'
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ================= 页面路由 ===================
@app.route('/')
@login_required
def index():
    return render_template('detailed_chat.html')

@app.route('/detailed_chat')
@login_required
def detailed_chat():
    return render_template('detailed_chat.html')

@app.route('/doctors')
@login_required
def doctors():
    return render_template('doctors.html')

@app.route('/doctor_chat/<int:doctor_id>')
@login_required
def doctor_chat_page(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    return render_template('doctor_chat.html', doctor=doctor)

# =============== API 路由 =================
@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    try:
        data = request.get_json()
        query = data.get('query', '')
        multimodal_inputs = data.get('multimodal_inputs', [])
        enable_thinking = data.get('enable_thinking', True)
        conversation_id = data.get('conversation_id')

        if not query and multimodal_inputs:
            if any(input_data['type'] == 'image' for input_data in multimodal_inputs):
                query = "请分析这张图片中的内容，如果是中药相关的图片，请提取其中的中药名称、症状描述等信息。如果是手写文字，请识别其中的文字内容。"
            elif any(input_data['type'] == 'audio' for input_data in multimodal_inputs):
                query = "请分析这段音频中的内容，提取其中的症状描述或相关信息。"
            else:
                query = "请分析上传的多模态内容。"

        if not query and not multimodal_inputs:
            return jsonify({'error': '请输入问题或上传文件'}), 400

        processed_inputs = []
        for input_data in multimodal_inputs:
            if input_data['type'] == 'text':
                processed = multimodal_processor.process_text(input_data['content'])
            elif input_data['type'] == 'image':
                processed = multimodal_processor.process_image(input_data['file_path'])
            elif input_data['type'] == 'audio':
                processed = multimodal_processor.process_audio(input_data['file_path'])
            else:
                continue
            if processed.get('processed'):
                processed_inputs.append(processed)

        if conversation_id:
            conv = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first()
            if not conv:
                return jsonify({'error': '会话不存在或无权访问'}), 403
        else:
            title = query[:50] + '...' if len(query) > 50 else query
            conv = Conversation(user_id=current_user.id, title=title)
            db.session.add(conv)
            db.session.flush()

        result = rag_engine.chat(query, processed_inputs)

        user_msg = Message(
            conversation_id=conv.id,
            role='user',
            content=query,
            multimodal_data=multimodal_inputs
        )
        db.session.add(user_msg)

        bot_msg = Message(
            conversation_id=conv.id,
            role='bot',
            content=result.get('answer', '')
        )
        db.session.add(bot_msg)

        db.session.commit()

        return jsonify({
            'success': True,
            'conversation_id': conv.id,
            'answer': result.get('answer', ''),
            'query': query,
            'analysis': result.get('analysis', {}),
            'step': result.get('step', 'unknown'),
            'retrieved_nodes': result.get('retrieved_nodes', []),
            'cypher_queries': result.get('cypher_queries', []),
            'query_results': result.get('query_results', []),
            'processing_time': result.get('processing_time', 0),
            'visualization': result.get('visualization', {}),
            'table_data': result.get('table_data', []),
            'multimodal_processed': processed_inputs
        })

    except Exception as e:
        db.session.rollback()
        print(f"API Chat 异常: {str(e)}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    conversations = Conversation.query.filter_by(user_id=current_user.id)\
                      .order_by(Conversation.created_at.desc()).all()
    result = []
    for conv in conversations:
        first_msg = Message.query.filter_by(conversation_id=conv.id, role='user')\
                     .order_by(Message.timestamp).first()
        title = first_msg.content[:30] + '...' if first_msg else '新对话'
        result.append({
            'id': conv.id,
            'title': title,
            'created_at': conv.created_at.isoformat() + 'Z'
        })
    return jsonify(result)

@app.route('/api/history/<int:conv_id>', methods=['GET'])
@login_required
def get_conversation(conv_id):
    conv = Conversation.query.get_or_404(conv_id)
    if conv.user_id != current_user.id:
        return jsonify({'error': '无权访问'}), 403
    messages = []
    for msg in conv.messages:
        messages.append({
            'role': msg.role,
            'content': msg.content,
            'multimodal_data': msg.multimodal_data
        })
    return jsonify(messages)

@app.route('/api/history/<int:conv_id>', methods=['DELETE'])
@login_required
def delete_conversation(conv_id):
    conv = Conversation.query.get_or_404(conv_id)
    if conv.user_id != current_user.id:
        return jsonify({'error': '无权访问'}), 403
    Message.query.filter_by(conversation_id=conv.id).delete()
    db.session.delete(conv)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)
            file_info = {
                'filename': filename,
                'file_path': file_path,
                'file_type': file.filename.rsplit('.', 1)[1].lower(),
                'size': os.path.getsize(file_path)
            }
            return jsonify({'success': True, 'file_info': file_info})
        return jsonify({'success': False, 'error': '不支持的文件类型'}), 400
    except Exception as e:
        print(f"文件上传异常: {e}")
        return jsonify({'success': False, 'error': f'上传文件时出错: {str(e)}'}), 500

@app.route('/api/process_multimodal', methods=['POST'])
def process_multimodal():
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        file_type = data.get('file_type', '')
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 400
        if file_type in ['png', 'jpg', 'jpeg', 'gif']:
            result = multimodal_processor.process_image(file_path)
        elif file_type in ['wav', 'mp3', 'm4a']:
            result = multimodal_processor.process_audio(file_path)
        else:
            return jsonify({'error': '不支持的文件类型'}), 400
        return jsonify({'success': True, 'processed_result': result})
    except Exception as e:
        return jsonify({'error': f'处理多模态输入时出错: {str(e)}'}), 500

@app.route('/api/doctors', methods=['GET'])
@login_required
def get_doctors():
    doctors = Doctor.query.filter_by(is_active=True).all()
    now = datetime.now().time()
    result = []
    for doc in doctors:
        if doc.work_start <= doc.work_end:
            is_available = doc.work_start <= now <= doc.work_end
        else:
            is_available = now >= doc.work_start or now <= doc.work_end
        result.append({
            'id': doc.id,
            'name': doc.name,
            'specialty': doc.specialty,
            'work_start': doc.work_start.strftime('%H:%M'),
            'work_end': doc.work_end.strftime('%H:%M'),
            'is_available': is_available,
            'description': doc.description
        })
    return jsonify(result)

# ================ WebSocket 事件（实时聊天） ================
@socketio.on('join_doctor_room')
def handle_join_doctor_room(data):
    user_id = data.get('user_id')
    if not user_id or int(user_id) != current_user.id:
        emit('error', {'message': '无权访问'})
        return
    doctor_id = data.get('doctor_id')
    room = f"doctor_{doctor_id}_user_{user_id}"
    join_room(room)

    # 加载历史消息
    messages = DoctorMessage.query.filter_by(
        doctor_id=doctor_id,
        user_id=user_id
    ).order_by(DoctorMessage.timestamp).all()
    for msg in messages:
        emit('new_message', {
            'sender': msg.sender_role,
            'message': msg.message,
            'timestamp': msg.timestamp.isoformat()
        }, room=room)

@socketio.on('send_message')
def handle_send_message(data):
    doctor_id = data.get('doctor_id')
    user_id = current_user.id
    message_text = data.get('message')
    room = f"doctor_{doctor_id}_user_{user_id}"

    # 保存用户消息到 DoctorMessage
    user_doc_msg = DoctorMessage(
        doctor_id=doctor_id,
        user_id=user_id,
        message=message_text,
        sender_role='user'
    )
    db.session.add(user_doc_msg)

    # 保存到统一对话历史
    doctor = Doctor.query.get(doctor_id)
    conv_title = f"与{doctor.name}的对话"
    conv = Conversation.query.filter_by(user_id=user_id, title=conv_title).first()
    if not conv:
        conv = Conversation(user_id=user_id, title=conv_title)
        db.session.add(conv)
        db.session.flush()
    user_msg = Message(
        conversation_id=conv.id,
        role='user',
        content=message_text
    )
    db.session.add(user_msg)
    db.session.commit()

    # 广播用户消息
    emit('new_message', {
        'sender': 'user',
        'message': message_text,
        'timestamp': user_doc_msg.timestamp.isoformat()
    }, room=room)

    # 生成医生回复
    reply = get_auto_reply(message_text, doctor_id)

    # 保存医生回复
    doctor_doc_msg = DoctorMessage(
        doctor_id=doctor_id,
        user_id=user_id,
        message=reply,
        sender_role='doctor'
    )
    db.session.add(doctor_doc_msg)
    doctor_msg = Message(
        conversation_id=conv.id,
        role='doctor',
        content=reply
    )
    db.session.add(doctor_msg)
    db.session.commit()

    # 广播医生回复
    emit('new_message', {
        'sender': 'doctor',
        'message': reply,
        'timestamp': doctor_doc_msg.timestamp.isoformat()
    }, room=room)

# =============== 数据库表创建与启动 =================
with app.app_context():
    db.create_all()

    # 初始化医生数据（如果为空）
    if Doctor.query.count() == 0:
        doctors_data = [
            {
                'name': '李时珍',
                'specialty': '中医全科，擅长内科、妇科',
                'work_start': time(8, 0),
                'work_end': time(12, 0),
                'description': '性格温和，耐心细致，擅长辨证论治。',
                'is_active': True
            },
            {
                'name': '张仲景',
                'specialty': '伤寒杂病，擅长经方应用',
                'work_start': time(14, 0),
                'work_end': time(18, 0),
                'description': '严谨认真，经验丰富，注重方证对应。',
                'is_active': True
            },
            {
                'name': '孙思邈',
                'specialty': '中医全科，擅长养生、针灸',
                'work_start': time(19, 0),
                'work_end': time(23, 0),
                'description': '博学多才，注重养生与针灸结合。',
                'is_active': True,
            }
        ]
        for d in doctors_data:
            doctor = Doctor(**d)
            db.session.add(doctor)
        db.session.commit()
        print("已添加默认医生信息")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)