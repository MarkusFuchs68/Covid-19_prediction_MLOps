from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ml_host_backend.app.exceptions.client_exceptions import InvalidArgumentException
from ml_host_backend.app.exceptions.service_exceptions import ModelNotFoundException
from ml_host_backend.app.routes.models import router as models_router

app = FastAPI()


@app.exception_handler(InvalidArgumentException)
async def handle_invalid_argument_eception(
    request: Request, exception: InvalidArgumentException
):
    return JSONResponse(status_code=400, content={"message": exception.message})


@app.exception_handler(ModelNotFoundException)
async def handle_model_not_found(request: Request, exception: ModelNotFoundException):
    return JSONResponse(status_code=404, content={"message": exception.message})


app.include_router(models_router, prefix="/api/models", tags=["models"])


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}
