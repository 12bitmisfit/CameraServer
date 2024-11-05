from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os


def create_app(video_config):
    app = FastAPI()

    streams = [stream['url'] for stream in video_config.get('streams', [])]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, 'web', 'index.html')

    @app.get("/", response_class=FileResponse)
    async def serve_html():
        return FileResponse(html_path)

    @app.get("/streams")
    async def get_streams():
        return JSONResponse(content=streams)

    return app


def web_server(video_config, base_url, port):
    app = create_app(video_config)
    uvicorn.run(app, host=base_url, port=port)
