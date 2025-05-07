from nest.core import PyNestFactory, Module
from fastapi.middleware.cors import CORSMiddleware

from .app_controller import AppController
from .app_service import AppService


@Module(imports=[], controllers=[AppController], providers=[AppService])
class AppModule:
    pass


app = PyNestFactory.create(
    AppModule,
    description="This is my COS40007 app.",
    title="COS40007 Application",
    version="1.0.0",
    debug=True,
)

# Add CORS middleware


http_server = app.get_server()

http_server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
