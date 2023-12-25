from flask import Flask
from flask_cors import CORS


from routes.route import api_v1

import settings

# Create App
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Register API
app.register_blueprint(api_v1, url_prefix='/ai')