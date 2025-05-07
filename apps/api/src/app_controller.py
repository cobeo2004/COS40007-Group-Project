from nest.core import Controller, Get, Post
from .app_service import AppService
from src.dto.test_dto import TestDto, TestDto2

@Controller("/")
class AppController:

    def __init__(self, service: AppService):
        self.service = service

    @Get("/", response_model=TestDto2)
    def say_hello(self):
        return {"message": "Hello, World!"}

    @Post("/", response_model=TestDto)
    def get_app_info(self, input_dto: TestDto):
        return self.service.get_app_info(input_dto)
