from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    pages: int
    chunks: int


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class ErrorResponse(BaseModel):
    detail: str