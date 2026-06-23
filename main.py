import os
import json
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from analyzer import ExcelAnalyzer
from ai_agent import AIAgent

load_dotenv()

app = FastAPI(title="Excel AI Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')

os.makedirs(UPLOAD_DIR, exist_ok=True)

ai_agent = AIAgent(api_key=OPENROUTER_API_KEY)
users = {}
sessions = {}


class QueryRequest(BaseModel):
    session_id: str
    query: str


class ChartRequest(BaseModel):
    session_id: str
    chart_type: str
    x_col: str
    y_col: str = None
    title: str = None


class GoogleLoginRequest(BaseModel):
    id_token: str


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))


@app.get("/{path:path}")
async def serve_static(path: str):
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))


@app.post("/api/auth/google")
async def google_login(req: GoogleLoginRequest):
    try:
        info = id_token.verify_oauth2_token(
            req.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        user_id = info['sub']
        existing = users.get(user_id, {})
        user_data = {
            'user_id': user_id,
            'email': info.get('email', ''),
            'name': info.get('name', ''),
            'picture': info.get('picture', ''),
        }
        user_data.update(existing)
        users[user_id] = user_data

        return {
            'success': True,
            'user': {
                'user_id': user_id,
                'email': user_data['email'],
                'name': user_data['name'],
                'picture': user_data['picture'],
            }
        }
    except ValueError as e:
        raise HTTPException(401, f"Invalid token: {str(e)}")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(400, "Only .xlsx, .xls, .csv files are supported")

    session_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")

    content = await file.read()
    with open(save_path, 'wb') as f:
        f.write(content)

    try:
        analyzer = ExcelAnalyzer(save_path)
        preview = analyzer.get_preview()
        sessions[session_id] = {
            'analyzer': analyzer,
            'file_path': save_path,
            'file_name': file.filename,
            'messages': [],
        }

        insights = ""
        if ai_agent.is_available():
            insights = ai_agent.generate_insights(preview['summary'], preview['head'])

        return {
            'session_id': session_id,
            'file_name': file.filename,
            'preview': preview,
            'insights': insights
        }
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(400, f"Could not process file: {str(e)}")


@app.post("/api/query")
async def process_query(req: QueryRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[req.session_id]
    analyzer = session['analyzer']
    summary = analyzer.summary

    code = ai_agent.generate_code(req.query, summary, analyzer.columns, analyzer.get_head(5))
    result = analyzer.execute_code(code)

    session['messages'].append({'role': 'user', 'content': req.query})
    session['messages'].append({'role': 'assistant', 'content': result})

    return {'result': result, 'history': session['messages'][-4:]}


@app.post("/api/chart")
async def create_chart(req: ChartRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    analyzer = sessions[req.session_id]['analyzer']
    chart = analyzer.generate_chart(req.chart_type, req.x_col, req.y_col, req.title)

    if not chart:
        raise HTTPException(400, "Could not generate chart")

    return chart


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]
    return {
        'session_id': session_id,
        'file_name': session['file_name'],
        'history': session['messages']
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
