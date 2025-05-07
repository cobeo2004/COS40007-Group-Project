from pydantic import BaseModel


class TestDto(BaseModel):
    name: str

class TestDto2(BaseModel):
    message: str
