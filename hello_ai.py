import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic, AuthenticationError, APIConnectionError

load_dotenv(override=True)
#override=True ensures that the environment variables in the .env file take precedence over any existing environment variables. 
#similar like model.refresh(true)
 
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("Error: ANTHROPIC_API_KEY not found in environment variables.")
    print("Copy .env.example to .env and add your key.")
    sys.exit(1)

client = Anthropic(api_key=api_key)
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

# Call block
prompt = "What is generative AI in one sentence?"
print("Sending prompt: '", prompt, "'")

try: 
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": prompt}
        ],
        extra_body={"temperature": 0.7}
    )
    print(f"Response: {response.content[0].text}")
    print(f"Model: {response.model}")
    print(f"Input tokens:  {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print(f"Total tokens:  {response.usage.input_tokens + response.usage.output_tokens}")


except AuthenticationError:
    print("Error: Invalid API key. Check your .env file.")
except APIConnectionError:
    print("Error: Unable to connect to the Anthropic API.")
except Exception as e:
    print("An unexpected error occurred:", str(e))