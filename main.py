import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to an AI model."
    )
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
        raise ValueError(
            "OPENROUTER_API_KEY is not set in the environment variables."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def generate_content(client: OpenAI, prompt: str) -> ChatCompletion:
    return client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )


def main() -> None:
    args = parse_arguments()

    with create_client() as client:
        response = generate_content(client, args.user_input)

    if not response.choices:
        raise RuntimeError("The API returned no choices.")

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The API returned no text content.")

    if args.verbose:
        print(f"User prompt: {args.user_input}")

        if response.usage is not None:
            print(f"Prompt tokens: {response.usage.prompt_tokens}")
            print(f"Response tokens: {response.usage.completion_tokens}")

    print("Generated Content:\n", content)


if __name__ == "__main__":
    main()