import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

def log_request(user_input):
    logging.info(f"{datetime.now()} - User input: {user_input}")

def format_response(response):
    # Implement any necessary formatting
    return response.strip()
