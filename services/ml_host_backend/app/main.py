from fastapi import FastAPI
from app.routes.models import router as models_router
from app.exceptions.service_exceptions import ModelNotFoundException
from app.exceptions.client_exceptions import InvalidArgumentException
from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(InvalidArgumentException)
async def handle_invalid_argument_eception(
    request: Request,
    exception: InvalidArgumentException
    ):
    print("here1")
    return JSONResponse(
        status_code=400,
        content={
            'message': exception.message
        }
    )
@app.exception_handler(ModelNotFoundException)
async def handle_model_not_found(
    request: Request,
    exception: ModelNotFoundException
    ):
    print("here")
    return JSONResponse(
        status_code=404,
        content={
            'message': exception.message
        }
    )

app.include_router(models_router, prefix="/api/models", tags=["models"])

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}
