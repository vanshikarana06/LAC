import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads your .env file and populates os.environ

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
for m in client.models.list().data:
    print(m.id)