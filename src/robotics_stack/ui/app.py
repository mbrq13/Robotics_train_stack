"""Operations API and dashboard."""

from __future__ import annotations

from pathlib import Path

from robotics_stack.services.station import RobotStation


def create_app(station: RobotStation):
    try:
        from fastapi import FastAPI, HTTPException, Response
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("install the ui extra to run the operations console") from exc

    app = FastAPI(title="Robotics Station", version="0.1.0")
    index = Path(__file__).resolve().parents[1] / "web" / "index.html"

    @app.get("/")
    def dashboard():
        return FileResponse(index)

    @app.get("/api/status")
    def status():
        return station.status().__dict__

    @app.get("/api/cameras/{name}.jpg")
    def camera(name: str):
        try:
            return Response(content=station.camera_frame(name), media_type="image/jpeg")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/validate")
    def validate():
        return _invoke(station.validate_hardware)

    @app.post("/api/arm")
    def arm():
        return _invoke(lambda: {"session_id": station.arm()})

    @app.post("/api/start")
    def start():
        return _invoke(lambda: (station.start(), {"state": station.status().state})[1])

    @app.post("/api/pause")
    def pause():
        return _invoke(lambda: (station.pause(), {"state": station.status().state})[1])

    @app.post("/api/home")
    def home():
        return _invoke(lambda: (station.home(), {"state": station.status().state})[1])

    @app.post("/api/reset-fault")
    def reset_fault():
        return _invoke(lambda: (station.clear_fault(), {"state": station.status().state})[1])

    def _invoke(operation):
        try:
            value = operation()
            return value or {"ok": True}
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app
