from nest.core import Injectable
from src.dto.test_dto import TestDto

@Injectable
class AppService:
    def __init__(self):
        self.app_name = "Pynest App"
        self.app_version = "1.0.0"

    def get_app_info(self, input_dto: TestDto) -> TestDto:
        return {"name": input_dto.name}

