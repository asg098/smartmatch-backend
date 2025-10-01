import os
import json
import hashlib
import uuid
import bcrypt
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from functools import wraps
from jose import jwt
from base64 import b64decode
import cv2
from fer import FER
from transformers import pipeline
from textblob import TextBlob
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)
app.config.update(SECRET_KEY='smartmatch_secret_2025', MAX_CONTENT_LENGTH=50*1024*1024)

detector = FER(mtcnn=True)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
users_db, profiles_db, assessments_db, interviews_db, applications_db, jobs_db, blockchain_db, interview_sessions = {}, {}, {}, {}, {}, {}, [], {}

QUESTION_BANK = {
    'Software Engineer': {
        'assessment': [
            {'id': 1, 'question': 'What is the time complexity of binary search?', 'options': ['O(n)', 'O(log n)', 'O(n^2)', 'O(1)'], 'correct': 1},
            {'id': 2, 'question': 'Which data structure uses LIFO?', 'options': ['Queue', 'Stack', 'Array', 'Tree'], 'correct': 1},
            {'id': 3, 'question': 'What is polymorphism in OOP?', 'options': ['Data hiding', 'Multiple forms', 'Inheritance', 'Encapsulation'], 'correct': 1},
            {'id': 4, 'question': 'What is a REST API?', 'options': ['Database', 'Architectural style', 'Programming language', 'Framework'], 'correct': 1},
            {'id': 5, 'question': 'What is Git used for?', 'options': ['Testing', 'Version control', 'Deployment', 'Database'], 'correct': 1},
            {'id': 6, 'question': 'What is Big O notation?', 'options': ['Algorithm efficiency', 'Data type', 'Variable', 'Function'], 'correct': 0},
            {'id': 7, 'question': 'What is a linked list?', 'options': ['Static array', 'Dynamic data structure', 'Hash table', 'Graph'], 'correct': 1},
            {'id': 8, 'question': 'What is inheritance in OOP?', 'options': ['Code reuse', 'Data hiding', 'Looping', 'Branching'], 'correct': 0},
            {'id': 9, 'question': 'What is SQL injection?', 'options': ['Feature', 'Security vulnerability', 'Database type', 'Query optimizer'], 'correct': 1},
            {'id': 10, 'question': 'What is asynchronous programming?', 'options': ['Sequential execution', 'Non-blocking operations', 'Error handling', 'Data storage'], 'correct': 1}
        ],
        'interview': [
            'Explain your experience with software development and programming languages.',
            'Describe a complex technical problem you solved.',
            'How do you approach code reviews and collaborative development?',
            'What is your testing and debugging methodology?',
            'How do you stay updated with new technologies and best practices?'
        ]
    },
    'Data Analyst': {
        'assessment': [
            {'id': 1, 'question': 'What does SQL stand for?', 'options': ['Structured Query Language', 'Simple Query Language', 'Standard Query Language', 'System Query Language'], 'correct': 0},
            {'id': 2, 'question': 'What is a primary key?', 'options': ['Foreign key', 'Unique identifier', 'Index', 'Column type'], 'correct': 1},
            {'id': 3, 'question': 'What is data visualization?', 'options': ['Data storage', 'Graphical representation', 'Data cleaning', 'Database query'], 'correct': 1},
            {'id': 4, 'question': 'What is pandas in Python?', 'options': ['Animal', 'Data analysis library', 'Database', 'Web framework'], 'correct': 1},
            {'id': 5, 'question': 'What does ETL stand for?', 'options': ['Extract Transform Load', 'Error Test Load', 'Easy Transfer Load', 'Export Test Load'], 'correct': 0},
            {'id': 6, 'question': 'What is a JOIN in SQL?', 'options': ['Combine tables', 'Delete records', 'Update data', 'Create table'], 'correct': 0},
            {'id': 7, 'question': 'What is data normalization?', 'options': ['Data backup', 'Organizing data', 'Data deletion', 'Data encryption'], 'correct': 1},
            {'id': 8, 'question': 'What is a dashboard?', 'options': ['Database', 'Visual data display', 'Query tool', 'Backup system'], 'correct': 1},
            {'id': 9, 'question': 'What is mean, median, mode?', 'options': ['Statistical measures', 'Database types', 'Programming languages', 'Data formats'], 'correct': 0},
            {'id': 10, 'question': 'What is data mining?', 'options': ['Data deletion', 'Pattern discovery', 'Data backup', 'Data entry'], 'correct': 1}
        ],
        'interview': [
            'Tell us about your data analysis experience and tools you use.',
            'How do you ensure data accuracy and quality?',
            'Describe a data-driven decision you influenced.',
            'How do you present complex data to non-technical stakeholders?',
            'What statistical methods are you comfortable with?'
        ]
    },
    'DevOps Engineer': {
        'assessment': [
            {'id': 1, 'question': 'What is Docker?', 'options': ['Virtual machine', 'Containerization platform', 'Programming language', 'Database'], 'correct': 1},
            {'id': 2, 'question': 'What is Kubernetes used for?', 'options': ['Container orchestration', 'Code editing', 'Database management', 'Testing'], 'correct': 0},
            {'id': 3, 'question': 'What does CI/CD stand for?', 'options': ['Continuous Integration/Deployment', 'Code Integration/Development', 'Cloud Integration/Database', 'Container Image/Docker'], 'correct': 0},
            {'id': 4, 'question': 'What is Infrastructure as Code?', 'options': ['Manual configuration', 'Automated infrastructure management', 'Cloud-only feature', 'Database management'], 'correct': 1},
            {'id': 5, 'question': 'What is AWS?', 'options': ['Database', 'Cloud computing platform', 'Programming language', 'Container tool'], 'correct': 1},
            {'id': 6, 'question': 'What is a load balancer?', 'options': ['Database', 'Distributes traffic', 'Storage system', 'Code editor'], 'correct': 1},
            {'id': 7, 'question': 'What is monitoring in DevOps?', 'options': ['Code review', 'System health tracking', 'Testing', 'Deployment'], 'correct': 1},
            {'id': 8, 'question': 'What is a pipeline?', 'options': ['Database', 'Automated workflow', 'Container', 'Virtual machine'], 'correct': 1},
            {'id': 9, 'question': 'What is Jenkins?', 'options': ['Database', 'CI/CD automation tool', 'Cloud platform', 'Container'], 'correct': 1},
            {'id': 10, 'question': 'What is microservices architecture?', 'options': ['Single application', 'Distributed services', 'Database type', 'Testing method'], 'correct': 1}
        ],
        'interview': [
            'Explain your experience with cloud platforms and containerization.',
            'How do you handle production incidents and troubleshooting?',
            'Describe your understanding of DevOps culture and practices.',
            'What is your experience with automation and scripting?',
            'How do you ensure system reliability and scalability?'
        ]
    },
    'Frontend Developer': {
        'assessment': [
            {'id': 1, 'question': 'What is React?', 'options': ['Framework', 'JavaScript library', 'Programming language', 'Database'], 'correct': 1},
            {'id': 2, 'question': 'What is CSS used for?', 'options': ['Logic', 'Styling', 'Database', 'Server'], 'correct': 1},
            {'id': 3, 'question': 'What is the DOM?', 'options': ['Database', 'Document Object Model', 'Programming language', 'Server'], 'correct': 1},
            {'id': 4, 'question': 'What is responsive design?', 'options': ['Fast loading', 'Adapts to screen sizes', 'Animation', 'Security'], 'correct': 1},
            {'id': 5, 'question': 'What is a promise in JavaScript?', 'options': ['Variable', 'Async operation result', 'Function', 'Array'], 'correct': 1},
            {'id': 6, 'question': 'What is state in React?', 'options': ['CSS property', 'Component data', 'HTML tag', 'Database'], 'correct': 1},
            {'id': 7, 'question': 'What is flexbox?', 'options': ['Database', 'CSS layout model', 'JavaScript library', 'HTML tag'], 'correct': 1},
            {'id': 8, 'question': 'What is webpack?', 'options': ['Database', 'Module bundler', 'CSS framework', 'Testing tool'], 'correct': 1},
            {'id': 9, 'question': 'What are hooks in React?', 'options': ['CSS', 'State management functions', 'HTML tags', 'Database'], 'correct': 1},
            {'id': 10, 'question': 'What is API integration?', 'options': ['CSS styling', 'Connecting to backend', 'HTML structure', 'Testing'], 'correct': 1}
        ],
        'interview': [
            'Explain your experience with frontend frameworks and libraries.',
            'How do you ensure cross-browser compatibility?',
            'Describe your approach to responsive web design.',
            'How do you optimize website performance?',
            'What is your experience with state management?'
        ]
    },
    'Backend Developer': {
        'assessment': [
            {'id': 1, 'question': 'What is an API?', 'options': ['Database', 'Application Programming Interface', 'Programming language', 'Frontend library'], 'correct': 1},
            {'id': 2, 'question': 'What is a database index?', 'options': ['Table', 'Performance optimization', 'Query', 'Column'], 'correct': 1},
            {'id': 3, 'question': 'What is authentication?', 'options': ['Database', 'User verification', 'Frontend design', 'Testing'], 'correct': 1},
            {'id': 4, 'question': 'What is a RESTful service?', 'options': ['Database', 'Web service architecture', 'Frontend framework', 'Testing tool'], 'correct': 1},
            {'id': 5, 'question': 'What is middleware?', 'options': ['Database', 'Request processing layer', 'Frontend component', 'Testing tool'], 'correct': 1},
            {'id': 6, 'question': 'What is caching?', 'options': ['Deleting data', 'Storing for quick access', 'Testing', 'Deployment'], 'correct': 1},
            {'id': 7, 'question': 'What is a web server?', 'options': ['Database', 'Handles HTTP requests', 'Frontend framework', 'Testing tool'], 'correct': 1},
            {'id': 8, 'question': 'What is NoSQL?', 'options': ['No database', 'Non-relational database', 'SQL replacement', 'Frontend library'], 'correct': 1},
            {'id': 9, 'question': 'What is JWT?', 'options': ['Database', 'JSON Web Token', 'Programming language', 'Frontend library'], 'correct': 1},
            {'id': 10, 'question': 'What is scalability?', 'options': ['Testing', 'Handling increased load', 'Design pattern', 'Database type'], 'correct': 1}
        ],
        'interview': [
            'Explain your backend development experience and technologies.',
            'How do you design and implement RESTful APIs?',
            'Describe your approach to database design and optimization.',
            'How do you handle security and authentication?',
            'What is your experience with scaling backend systems?'
        ]
    },
    'Digital Marketing': {
        'assessment': [
            {'id': 1, 'question': 'What is SEO?', 'options': ['Social media', 'Search Engine Optimization', 'Email marketing', 'Content writing'], 'correct': 1},
            {'id': 2, 'question': 'What is PPC?', 'options': ['Pay Per Click', 'Page Per Content', 'Post Per Comment', 'People Per Content'], 'correct': 0},
            {'id': 3, 'question': 'What is CTR?', 'options': ['Cost To Revenue', 'Click Through Rate', 'Content To Reader', 'Call To Read'], 'correct': 1},
            {'id': 4, 'question': 'What is content marketing?', 'options': ['Only ads', 'Valuable content creation', 'Email only', 'Social media only'], 'correct': 1},
            {'id': 5, 'question': 'What is A/B testing?', 'options': ['Grading system', 'Comparing versions', 'Content writing', 'Social posting'], 'correct': 1},
            {'id': 6, 'question': 'What is Google Analytics?', 'options': ['Social media', 'Web analytics tool', 'Email service', 'Ad platform'], 'correct': 1},
            {'id': 7, 'question': 'What is engagement rate?', 'options': ['Revenue', 'Audience interaction metric', 'Cost metric', 'Time metric'], 'correct': 1},
            {'id': 8, 'question': 'What is influencer marketing?', 'options': ['Traditional ads', 'Partnerships with influencers', 'Email only', 'SEO only'], 'correct': 1},
            {'id': 9, 'question': 'What is conversion rate?', 'options': ['Revenue', 'Percentage completing goal', 'Click rate', 'View rate'], 'correct': 1},
            {'id': 10, 'question': 'What is email marketing?', 'options': ['Social media', 'Email campaigns', 'SEO', 'Content writing'], 'correct': 1}
        ],
        'interview': [
            'Tell us about your digital marketing experience and campaigns.',
            'How do you measure marketing campaign success?',
            'Describe your approach to content strategy.',
            'How do you stay updated with digital marketing trends?',
            'What tools and platforms are you experienced with?'
        ]
    },
    'HR Manager': {
        'assessment': [
            {'id': 1, 'question': 'What is recruitment?', 'options': ['Training', 'Hiring process', 'Payroll', 'Performance review'], 'correct': 1},
            {'id': 2, 'question': 'What is onboarding?', 'options': ['Firing', 'New employee integration', 'Training', 'Recruitment'], 'correct': 1},
            {'id': 3, 'question': 'What is employee engagement?', 'options': ['Salary', 'Employee commitment', 'Hiring', 'Firing'], 'correct': 1},
            {'id': 4, 'question': 'What is performance appraisal?', 'options': ['Hiring', 'Employee evaluation', 'Payroll', 'Training'], 'correct': 1},
            {'id': 5, 'question': 'What is diversity and inclusion?', 'options': ['Training', 'Workplace equality', 'Payroll', 'Recruitment only'], 'correct': 1},
            {'id': 6, 'question': 'What is HRIS?', 'options': ['HR department', 'HR Information System', 'Hiring process', 'Training program'], 'correct': 1},
            {'id': 7, 'question': 'What is employee retention?', 'options': ['Firing', 'Keeping employees', 'Hiring', 'Training'], 'correct': 1},
            {'id': 8, 'question': 'What is conflict resolution?', 'options': ['Hiring', 'Resolving disputes', 'Payroll', 'Training'], 'correct': 1},
            {'id': 9, 'question': 'What is talent management?', 'options': ['Payroll', 'Developing employees', 'Firing', 'Recruitment only'], 'correct': 1},
            {'id': 10, 'question': 'What is company culture?', 'options': ['Office location', 'Organizational values', 'Salary structure', 'Training program'], 'correct': 1}
        ],
        'interview': [
            'Tell us about your HR experience and areas of expertise.',
            'How do you handle employee conflicts and grievances?',
            'Describe your approach to talent acquisition.',
            'How do you measure employee satisfaction and engagement?',
            'What is your experience with HR policies and compliance?'
        ]
    },
    'Product Manager': {
        'assessment': [
            {'id': 1, 'question': 'What is product roadmap?', 'options': ['Map', 'Product strategy timeline', 'Marketing plan', 'Sales plan'], 'correct': 1},
            {'id': 2, 'question': 'What is user story?', 'options': ['Novel', 'Feature from user perspective', 'Bug report', 'Design'], 'correct': 1},
            {'id': 3, 'question': 'What is MVP?', 'options': ['Most Valuable Product', 'Minimum Viable Product', 'Maximum Value Product', 'Market Value Product'], 'correct': 1},
            {'id': 4, 'question': 'What is product lifecycle?', 'options': ['Development only', 'Product stages', 'Marketing only', 'Sales only'], 'correct': 1},
            {'id': 5, 'question': 'What is market research?', 'options': ['Sales', 'Understanding market', 'Development', 'Design'], 'correct': 1},
            {'id': 6, 'question': 'What is stakeholder management?', 'options': ['Finance', 'Managing expectations', 'Development', 'Design'], 'correct': 1},
            {'id': 7, 'question': 'What is A/B testing?', 'options': ['Grading', 'Version comparison', 'Development', 'Design'], 'correct': 1},
            {'id': 8, 'question': 'What is user persona?', 'options': ['Employee', 'User profile representation', 'Product feature', 'Marketing plan'], 'correct': 1},
            {'id': 9, 'question': 'What is product metrics?', 'options': ['Size', 'Performance measurements', 'Price', 'Color'], 'correct': 1},
            {'id': 10, 'question': 'What is agile methodology?', 'options': ['Fixed plan', 'Iterative development', 'Marketing strategy', 'Sales process'], 'correct': 1}
        ],
        'interview': [
            'Tell us about your product management experience.',
            'How do you prioritize features and requirements?',
            'Describe your approach to user research and validation.',
            'How do you work with cross-functional teams?',
            'What metrics do you use to measure product success?'
        ]
    }
}

JOB_CATEGORIES = list(QUESTION_BANK.keys())

def generate_token(uid, role):
    return jwt.encode({'user_id': uid, 'role': role, 'exp': datetime.utcnow()+timedelta(days=7)}, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token: return jsonify({'error': 'Token missing'}), 401
        try:
            token = token.split(' ')[1] if ' ' in token else token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id, request.user_role = data['user_id'], data['role']
        except: return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

def add_blockchain(action, uid, data):
    block = {'id': len(blockchain_db), 'timestamp': datetime.utcnow().isoformat(), 'action': action, 'user_id': uid, 'data': data, 'hash': hashlib.sha256(f"{len(blockchain_db)}{datetime.utcnow().isoformat()}{action}{uid}".encode()).hexdigest()}
    blockchain_db.append(block)
    return block

def analyze_interview_response(text):
    blob = TextBlob(text)
    words = text.split()
    return {
        'word_count': len(words),
        'sentiment_polarity': blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity,
        'clarity_score': min(100, len(words) / 2),
        'keywords': [word for word in words if len(word) > 5][:5]
    }

@app.route('/api/auth/register', methods=['POST'])
def register():
    d = request.json
    email, pwd, role, name = d.get('email'), d.get('password'), d.get('role', 'student'), d.get('name', '')
    if not email or not pwd: return jsonify({'error': 'Email and password required'}), 400
    if email in users_db: return jsonify({'error': 'User exists'}), 400
    uid = str(uuid.uuid4())
    users_db[email] = {'id': uid, 'email': email, 'password': bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()), 'role': role, 'name': name, 'created_at': datetime.utcnow().isoformat()}
    add_blockchain('USER_REGISTERED', uid, {'email': email, 'role': role})
    return jsonify({'token': generate_token(uid, role), 'user': {'id': uid, 'email': email, 'role': role, 'name': name}}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    d = request.json
    email, pwd = d.get('email'), d.get('password')
    if not email or not pwd: return jsonify({'error': 'Email and password required'}), 400
    user = users_db.get(email)
    if not user or not bcrypt.checkpw(pwd.encode(), user['password']): return jsonify({'error': 'Invalid credentials'}), 401
    return jsonify({'token': generate_token(user['id'], user['role']), 'user': {'id': user['id'], 'email': email, 'role': user['role'], 'name': user['name']}}), 200

@app.route('/api/profile', methods=['POST'])
@verify_token
def create_profile():
    d = request.json
    profiles_db[request.user_id] = {'user_id': request.user_id, 'name': d.get('name',''), 'skills': d.get('skills',[]), 'education': d.get('education',''), 'experience': d.get('experience',''), 'location': d.get('location',''), 'category': d.get('category',''), 'resume': d.get('resume',''), 'created_at': datetime.utcnow().isoformat()}
    add_blockchain('PROFILE_CREATED', request.user_id, {'skills': d.get('skills',[])})
    return jsonify({'message': 'Profile created', 'profile': profiles_db[request.user_id]}), 201

@app.route('/api/profile', methods=['GET'])
@verify_token
def get_profile():
    p = profiles_db.get(request.user_id)
    return (jsonify({'profile': p}), 200) if p else (jsonify({'error': 'Profile not found'}), 404)

@app.route('/api/question-bank/categories', methods=['GET'])
@verify_token
def get_categories():
    if request.user_role != 'recruiter':
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify({'categories': JOB_CATEGORIES}), 200

@app.route('/api/question-bank/<category>', methods=['GET'])
@verify_token
def get_questions_by_category(category):
    if request.user_role != 'recruiter':
        return jsonify({'error': 'Unauthorized'}), 403
    if category not in QUESTION_BANK:
        return jsonify({'error': 'Category not found'}), 404
    return jsonify({
        'category': category,
        'assessment_questions': QUESTION_BANK[category]['assessment'],
        'interview_questions': QUESTION_BANK[category]['interview']
    }), 200

@app.route('/api/jobs', methods=['POST'])
@verify_token
def create_job():
    if request.user_role != 'recruiter':
        return jsonify({'error': 'Unauthorized'}), 403
    d = request.json

    if not d.get('company') or not d.get('position') or not d.get('category'):
        return jsonify({'error': 'Company, position, and category are required'}), 400

    category = d.get('category')
    if category not in QUESTION_BANK:
        return jsonify({'error': 'Invalid category'}), 400

    assessment_questions = QUESTION_BANK[category]['assessment']
    interview_questions = QUESTION_BANK[category]['interview']

    if d.get('custom_assessment_questions'):
        assessment_questions = d.get('custom_assessment_questions')
    if d.get('custom_interview_questions'):
        interview_questions = d.get('custom_interview_questions')

    jid = str(uuid.uuid4())
    jobs_db[jid] = {
        'id': jid,
        'company': d.get('company',''),
        'position': d.get('position',''),
        'category': category,
        'location': d.get('location',''),
        'required_skills': d.get('required_skills',[]),
        'description': d.get('description',''),
        'min_assessment_score': d.get('min_assessment_score', 60),
        'min_interview_score': d.get('min_interview_score', 65),
        'assessment_questions': assessment_questions,
        'interview_questions': interview_questions,
        'recruiter_id': request.user_id,
        'created_at': datetime.utcnow().isoformat()
    }
    add_blockchain('JOB_CREATED', request.user_id, {'job_id': jid, 'position': d.get('position',''), 'category': category})
    return jsonify({'message': 'Job created successfully', 'job_id': jid}), 201

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    return jsonify({'jobs': list(jobs_db.values())}), 200

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_detail(job_id):
    if job_id not in jobs_db:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({'job': jobs_db[job_id]}), 200

@app.route('/api/jobs/recruiter', methods=['GET'])
@verify_token
def get_recruiter_jobs():
    if request.user_role != 'recruiter':
        return jsonify({'error': 'Unauthorized'}), 403
    recruiter_jobs = [j for j in jobs_db.values() if j['recruiter_id'] == request.user_id]
    return jsonify({'jobs': recruiter_jobs}), 200

@app.route('/api/jobs/match', methods=['POST'])
@verify_token
def match_jobs():
    d = request.json
    user_skills = set([s.lower().strip() for s in d.get('skills', [])])
    if request.user_id not in profiles_db:
        return jsonify({'error': 'Profile not found'}), 404
    matches = []
    for jid, job in jobs_db.items():
        job_skills = set([s.lower().strip() for s in job['required_skills']])
        mc = len(user_skills.intersection(job_skills))
        if mc > 0:
            matches.append({
                'job_id': jid,
                'company': job['company'],
                'position': job['position'],
                'category': job.get('category', ''),
                'location': job['location'],
                'required_skills': job['required_skills'],
                'description': job.get('description', ''),
                'min_assessment_score': job.get('min_assessment_score', 0),
                'min_interview_score': job.get('min_interview_score', 0),
                'match_score': round((mc/len(job_skills))*100, 2) if len(job_skills) > 0 else 0
            })
    matches.sort(key=lambda x: x['match_score'], reverse=True)
    add_blockchain('JOB_MATCHED', request.user_id, {'matches': len(matches)})
    return jsonify({'matches': matches}), 200

@app.route('/api/jobs/apply', methods=['POST'])
@verify_token
def apply_job():
    jid = request.json.get('job_id')
    if not jid or jid not in jobs_db:
        return jsonify({'error': 'Job not found'}), 404

    for app in applications_db.values():
        if app['user_id'] == request.user_id and app['job_id'] == jid:
            return jsonify({'error': 'Already applied to this job'}), 400

    aid = str(uuid.uuid4())
    applications_db[aid] = {
        'id': aid,
        'user_id': request.user_id,
        'job_id': jid,
        'status': 'pending',
        'applied_at': datetime.utcnow().isoformat(),
        'assessment_completed': False,
        'interview_completed': False,
        'assessment_score': 0,
        'interview_score': 0
    }
    add_blockchain('JOB_APPLIED', request.user_id, {'job_id': jid})
    return jsonify({'message': 'Application submitted', 'application_id': aid}), 201

@app.route('/api/applications/my', methods=['GET'])
@verify_token
def get_my_applications():
    apps = []
    for a in applications_db.values():
        if a['user_id'] == request.user_id:
            job = jobs_db.get(a['job_id'], {})
            apps.append({
                'application_id': a['id'],
                'job_id': a['job_id'],
                'company': job.get('company', ''),
                'position': job.get('position', ''),
                'category': job.get('category', ''),
                'status': a['status'],
                'assessment_completed': a.get('assessment_completed', False),
                'interview_completed': a.get('interview_completed', False),
                'assessment_score': a.get('assessment_score', 0),
                'interview_score': a.get('interview_score', 0),
                'applied_at': a['applied_at']
            })
    apps.sort(key=lambda x: x['applied_at'], reverse=True)
    return jsonify({'applications': apps}), 200

@app.route('/api/assessment/start', methods=['POST'])
@verify_token
def start_assessment():
    d = request.json
    app_id = d.get('application_id')

    if not app_id or app_id not in applications_db:
        return jsonify({'error': 'Application not found'}), 404

    app = applications_db[app_id]
    if app['user_id'] != request.user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    if app['assessment_completed']:
        return jsonify({'error': 'Assessment already completed'}), 400

    job = jobs_db.get(app['job_id'])
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    questions = job.get('assessment_questions', [])

    aid = str(uuid.uuid4())
    assessments_db[aid] = {
        'id': aid,
        'user_id': request.user_id,
        'application_id': app_id,
        'job_id': app['job_id'],
        'questions': questions,
        'answers': {},
        'started_at': datetime.utcnow().isoformat(),
        'completed': False
    }

    return jsonify({'assessment_id': aid, 'questions': questions}), 200

@app.route('/api/assessment/submit', methods=['POST'])
@verify_token
def submit_assessment():
    d = request.json
    aid, ans = d.get('assessment_id'), d.get('answers', {})

    if not aid or aid not in assessments_db:
        return jsonify({'error': 'Assessment not found'}), 404

    a = assessments_db[aid]
    if a['user_id'] != request.user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    if a['completed']:
        return jsonify({'error': 'Assessment already submitted'}), 400

    score = sum(1 for q in a['questions'] if str(q['id']) in ans and ans[str(q['id'])] == q['correct'])
    total = len(a['questions'])
    pct = round((score/total)*100, 2)

    a.update({
        'answers': ans,
        'score': score,
        'total': total,
        'percentage': pct,
        'completed': True,
        'completed_at': datetime.utcnow().isoformat()
    })

    add_blockchain('ASSESSMENT_COMPLETED', request.user_id, {'job_id': a['job_id'], 'score': pct})

    if a.get('application_id') and a['application_id'] in applications_db:
        applications_db[a['application_id']].update({
            'assessment_completed': True,
            'assessment_score': pct
        })

    return jsonify({
        'score': score,
        'total': total,
        'percentage': pct,
        'message': 'Assessment completed successfully'
    }), 200

@app.route('/api/assessment/history', methods=['GET'])
@verify_token
def assessment_history():
    history = [a for a in assessments_db.values() if a['user_id'] == request.user_id and a['completed']]
    return jsonify({'assessments': history}), 200

@app.route('/api/interview/start', methods=['POST'])
@verify_token
def start_interview():
    d = request.json
    app_id = d.get('application_id')

    if not app_id or app_id not in applications_db:
        return jsonify({'error': 'Application not found'}), 404

    app = applications_db[app_id]
    if app['user_id'] != request.user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    if app['interview_completed']:
        return jsonify({'error': 'Interview already completed'}), 400

    job = jobs_db.get(app['job_id'])
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    questions = job.get('interview_questions', [])

    sid = str(uuid.uuid4())
    interview_sessions[sid] = {
        'id': sid,
        'user_id': request.user_id,
        'application_id': app_id,
        'job_id': app['job_id'],
        'questions': questions,
        'current_question': 0,
        'responses': [],
        'emotions': [],
        'confidence_scores': [],
        'started_at': datetime.utcnow().isoformat(),
        'completed': False,
        'video_writer': cv2.VideoWriter(f'interview_{sid}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (640,480))
    }

    return jsonify({'session_id': sid, 'question': questions[0], 'total_questions': len(questions)}), 200

@app.route('/api/interview/frame', methods=['POST'])
@verify_token
def process_interview_frame():
    d = request.json
    sid = d.get('session_id')

    if not sid or sid not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404

    s = interview_sessions[sid]
    if s['user_id'] != request.user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        frame = cv2.imdecode(np.frombuffer(b64decode(d.get('image')), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400
    except:
        return jsonify({'error': 'Invalid image data'}), 400

    de, es = 'neutral', 0.0
    try:
        res = detector.detect_emotions(frame)
        if res:
            ems = res[0]['emotions']
            de = max(ems, key=ems.get)
            es = ems[de]
            x, y, w, h = res[0]['box']
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{de}: {es:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    except:
        pass

    conf = es if de in ['happy','neutral'] else 1-es
    s['emotions'].append({'emotion': de, 'score': es})
    s['confidence_scores'].append(conf)

    cv2.putText(frame, f"Q: {s['questions'][s['current_question']][:40]}...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Confidence: {conf:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    s['video_writer'].write(frame)

    return jsonify({'emotion': de, 'confidence': conf, 'question_index': s['current_question']}), 200

@app.route('/api/interview/answer', methods=['POST'])
@verify_token
def process_interview_answer():
    d = request.json
    sid = d.get('session_id')

    if not sid or sid not in interview_sessions:
        return jsonify({'error': 'Session not found'}), 404

    s = interview_sessions[sid]
    if s['user_id'] != request.user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    atxt = d.get('answer', '').strip()

    if not atxt:
        return jsonify({'error': 'Answer cannot be empty'}), 400

    sent = sentiment_analyzer(atxt)[0]
    analysis = analyze_interview_response(atxt)

    s['responses'].append({
        'question': s['questions'][s['current_question']],
        'answer': atxt,
        'sentiment': sent['label'],
        'sentiment_score': sent['score'],
        'word_count': analysis['word_count'],
        'clarity': analysis['clarity_score'],
        'polarity': analysis['sentiment_polarity'],
        'keywords': analysis['keywords']
    })

    s['current_question'] += 1

    if s['current_question'] >= len(s['questions']):
        s['completed'] = True
        s['completed_at'] = datetime.utcnow().isoformat()
        s['video_writer'].release()

        ac = np.mean(s['confidence_scores']) if s['confidence_scores'] else 0.5
        ss = sum(1 for r in s['responses'] if r['sentiment']=='POSITIVE')/len(s['responses']) if s['responses'] else 0
        avg_clarity = np.mean([r['clarity'] for r in s['responses']]) / 100 if s['responses'] else 0
        avg_word_count = np.mean([r['word_count'] for r in s['responses']]) if s['responses'] else 0
        word_score = min(1.0, avg_word_count / 50)

        fs = round((ac*0.3 + ss*0.3 + avg_clarity*0.2 + word_score*0.2)*100, 2)
        s['final_score'] = fs
        s['detailed_analysis'] = {
            'avg_confidence': round(ac*100, 2),
            'positive_sentiment_rate': round(ss*100, 2),
            'avg_clarity': round(avg_clarity*100, 2),
            'avg_word_count': round(avg_word_count, 0),
            'total_questions': len(s['questions']),
            'emotions_detected': list(set([e['emotion'] for e in s['emotions']]))
        }

        interviews_db[sid] = s
        add_blockchain('INTERVIEW_COMPLETED', request.user_id, {'job_id': s['job_id'], 'score': fs})

        if s.get('application_id') and s['application_id'] in applications_db:
            applications_db[s['application_id']].update({
                'interview_completed': True,
                'interview_score': fs,
                'status': 'completed'
            })

        return jsonify({
            'completed': True,
            'score': fs,
            'analysis': s['detailed_analysis'],
            'message': 'Interview completed successfully'
        }), 200

    return jsonify({
        'completed': False,
        'next_question': s['questions'][s['current_question']],
        'question_index': s['current_question'],
        'total_questions': len(s['questions'])
    }), 200

@app.route('/api/interview/history', methods=['GET'])
@verify_token
def interview_history():
    history = [{k:v for k,v in i.items() if k!='video_writer'} for i in interviews_db.values() if i['user_id']==request.user_id and i['completed']]
    return jsonify({'interviews': history}), 200

@app.route('/api/recruiter/applications', methods=['GET'])
@verify_token
def get_applications():
    if request.user_role != 'recruiter':
        return jsonify({'error': 'Unauthorized'}), 403

    apps = []
    for a in applications_db.values():
        job = jobs_db.get(a['job_id'], {})

        if job.get('recruiter_id') != request.user_id:
            continue

        profile = profiles_db.get(a['user_id'], {})
        user = next((u for u in users_db.values() if u['id'] == a['user_id']), {})

        meets_requirements = True
        min_assessment = job.get('min_assessment_score', 0)
        min_interview = job.get('min_interview_score', 0)

        if a.get('assessment_score', 0) < min_assessment or a.get('interview_score', 0) < min_interview:
            meets_requirements = False

        apps.append({
            'application_id': a['id'],
            'candidate_name': profile.get('name', user.get('name', 'Unknown')),
            'candidate_email': user.get('email', ''),
            'candidate_skills': profile.get('skills', []),
            'position': job.get('position', 'Unknown'),
            'company': job.get('company', ''),
            'category': job.get('category', ''),
            'status': a['status'],
            'assessment_score': a.get('assessment_score', 0),
            'interview_score': a.get('interview_score', 0),
            'applied_at': a['applied_at'],
            'assessment_completed': a.get('assessment_completed', False),
            'interview_completed': a.get('interview_completed', False),
            'meets_requirements': meets_requirements,
            'min_assessment_required': min_assessment,
            'min_interview_required': min_interview
        })

    apps.sort(key=lambda x: x['assessment_score'] + x['interview_score'], reverse=True)

    return jsonify({'applications': apps[:10] if len(apps) > 10 else apps}), 200

@app.route('/api/recruiter/candidate/<app_id>', methods=['GET'])
@verify_token
def get_candidate_detail(app_id):
    if request.user_role != 'recruiter':
        return jsonify({'error': 'Unauthorized'}), 403

    if app_id not in applications_db:
        return jsonify({'error': 'Application not found'}), 404

    app = applications_db[app_id]
    profile = profiles_db.get(app['user_id'], {})
    user = next((u for u in users_db.values() if u['id'] == app['user_id']), {})

    assessments = [a for a in assessments_db.values() if a['user_id'] == app['user_id'] and a.get('application_id') == app_id]
    interviews = [{k:v for k,v in i.items() if k!='video_writer'} for i in interviews_db.values() if i['user_id'] == app['user_id'] and i.get('application_id') == app_id]

    return jsonify({
        'candidate': {
            'name': profile.get('name', user.get('name', 'Unknown')),
            'email': user.get('email', ''),
            'skills': profile.get('skills', []),
            'education': profile.get('education', ''),
            'experience': profile.get('experience', ''),
            'location': profile.get('location', '')
        },
        'application': app,
        'assessments': assessments,
        'interviews': interviews
    }), 200

@app.route('/api/recruiter/shortlist', methods=['POST'])
@verify_token
def shortlist_candidate():
    if request.user_role != 'recruiter':
        return jsonify({'error': 'Unauthorized'}), 403

    aid = request.json.get('application_id')
    if aid not in applications_db:
        return jsonify({'error': 'Application not found'}), 404

    applications_db[aid]['status'] = 'shortlisted'
    add_blockchain('CANDIDATE_SHORTLISTED', applications_db[aid]['user_id'], {'application_id': aid})

    return jsonify({'message': 'Candidate shortlisted successfully'}), 200

@app.route('/api/blockchain', methods=['GET'])
@verify_token
def get_blockchain():
    return jsonify({'blocks': [b for b in blockchain_db if b['user_id']==request.user_id]}), 200

@app.route('/api/blockchain/all', methods=['GET'])
@verify_token
def get_all_blockchain():
    if request.user_role != 'recruiter':
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify({'blocks': blockchain_db}), 200

@app.route('/api/stats', methods=['GET'])
@verify_token
def get_stats():
    if request.user_role == 'student':
        apps = [a for a in applications_db.values() if a['user_id']==request.user_id]
        asmt = [a for a in assessments_db.values() if a['user_id']==request.user_id and a['completed']]
        invs = [i for i in interviews_db.values() if i['user_id']==request.user_id and i['completed']]
        return jsonify({
            'applications': len(apps),
            'assessments_completed': len(asmt),
            'interviews_completed': len(invs),
            'avg_assessment_score': round(np.mean([a['percentage'] for a in asmt]), 2) if asmt else 0,
            'avg_interview_score': round(np.mean([i['final_score'] for i in invs]), 2) if invs else 0
        }), 200
    else:
        recruiter_jobs = [j for j in jobs_db.values() if j['recruiter_id']==request.user_id]
        all_apps = [a for a in applications_db.values() if jobs_db.get(a['job_id'], {}).get('recruiter_id')==request.user_id]
        return jsonify({
            'total_applications': len(all_apps),
            'total_jobs': len(recruiter_jobs),
            'shortlisted': len([a for a in all_apps if a['status']=='shortlisted']),
            'completed': len([a for a in all_apps if a['status']=='completed'])
        }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()}), 200

if __name__ == '__main__':
    ngrok.set_auth_token("33O2e1v6nr6rlU95KbpK7TMSclf_4gdejfL7nm3ySsUHPGRw9")
    public_url = ngrok.connect(5000).public_url
    print(f"Public URL: {public_url}")
    app.run(port=5000)