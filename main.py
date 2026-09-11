import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion
from prompts import get_system_prompt
from functions.call_function import call_function, get_available_functions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a prompt to an AI model.")
    parser.add_argument(
        "user_input",
        help="The prompt to send to the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the user prompt and token usage.",
    )
    return parser.parse_args()


def create_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def generate_content(client: OpenAI, messages: list) -> ChatCompletion:
    return client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages,
        tools=get_available_functions(),
        max_tokens=1024,
    )


MAX_ITERATIONS = 20


def main() -> None:
    args = parse_arguments()

    if args.verbose:
        print(f"User prompt: {args.user_input}")

    messages: list = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": args.user_input},
    ]

    with create_client() as client:
        for _ in range(MAX_ITERATIONS):
            response = generate_content(client, messages)

            if not response.choices:
                raise RuntimeError("The API returned no choices.")

            if args.verbose and response.usage is not None:
                print(f"Prompt tokens: {response.usage.prompt_tokens}")
                print(f"Response tokens: {response.usage.completion_tokens}")

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                print(message.content)
                return

            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    continue
                result_message = call_function(tool_call, verbose=args.verbose)
                if not result_message.get("content"):
                    raise RuntimeError(
                        f"No content returned for function: {tool_call.function.name}"
                    )
                if args.verbose:
                    print(f"-> {result_message['content']}")
                messages.append(result_message)

    print(f"Reached maximum iterations ({MAX_ITERATIONS}) without a final response.")
    sys.exit(1)


if __name__ == "__main__":
    main()
