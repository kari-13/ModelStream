import re
import sys

import pyperclip
from ollama import chat
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def extract_and_copy_code(text: str):
    """Extracts the first Python code block and copies it to the clipboard."""
    code_blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    if code_blocks:
        pyperclip.copy(code_blocks[0].strip())
        console.print(
            "\n[bold green]✔ First Python code snippet automatically copied to clipboard![/bold green]"
        )
    else:
        generic_blocks = re.findall(r"```\n?(.*?)\n```", text, re.DOTALL)
        if generic_blocks:
            pyperclip.copy(generic_blocks[0].strip())
            console.print(
                "\n[bold green]✔ Code snippet automatically copied to clipboard![/bold green]"
            )


def ask_ai(prompt: str):
    stream = chat(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_response = ""

    print("\n" + "═" * 60)

    # Initialize the Live context with an empty layout group
    with Live(
        Panel("Initializing...", title="Ollama"), refresh_per_second=15, console=console
    ) as live:
        for chunk in stream:
            content = chunk.message.content or ""
            if not content:
                continue

            full_response += content

            # Separate thinking process and actual response
            thinking_text = ""
            response_text = full_response

            if "thinking process:" in full_response.lower():
                parts = re.split(
                    r"thinking process:", full_response, flags=re.IGNORECASE
                )
                thinking_text = parts[0].strip()
                if len(parts) > 1:
                    # Look for actual markdown start or fallback to the rest of the text
                    content_match = re.search(r"(\n# .*|\n```.*)", parts[1], re.DOTALL)
                    response_text = (
                        content_match.group(1).strip()
                        if content_match
                        else parts[1].strip()
                    )
            elif (
                "<thinking>" in full_response.lower()
            ):  # Fallback for structural thinking tags
                parts = re.split(r"</thinking>", full_response, flags=re.IGNORECASE)
                thinking_text = parts[0].replace("<thinking>", "").strip()
                if len(parts) > 1:
                    response_text = parts[1].strip()

            # Build a clean UI using panels inside a display Group
            ui_components = []

            if thinking_text:
                ui_components.append(
                    Panel(
                        thinking_text,
                        title="[yellow]Thinking Process[/yellow]",
                        border_style="yellow",
                    )
                )

            if response_text:
                ui_components.append(
                    Panel(
                        Markdown(response_text),
                        title="[magenta]AI Response[/magenta]",
                        border_style="magenta",
                    )
                )

            # Update the live display with the structured container group
            if ui_components:
                live.update(Group(*ui_components))
            else:
                live.update(Panel(full_response, title="Streaming..."))

    # Extract code from the final processed response block
    extract_and_copy_code(
        response_text if "response_text" in locals() else full_response
    )


if __name__ == "__main__":
    console.print(
        "[bold green]Ollama Chat Initialized. Type 'quit' or 'bye' to exit.[/bold green]\n"
    )

    try:
        while True:
            user_prompt = input("\033[1;34mYou:\033[0m ")

            if user_prompt.strip().lower() in ["quit", "bye"]:
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break

            if not user_prompt.strip():
                continue

            ask_ai(user_prompt)
            print("\n" + "─" * 60 + "\n")

    except KeyboardInterrupt, EOFError:
        console.print("\n\n[bold red]Session ended.[/bold red]")
        sys.exit(0)
