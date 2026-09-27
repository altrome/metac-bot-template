import os

import dotenv

# Load environment variables once at module level.
# override=True: the .env file is the source of truth, so a stale exported
# variable (e.g. an old OPENAI_API_KEY in the shell profile) cannot shadow it.
dotenv.load_dotenv(override=True)

# Environment variables
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")