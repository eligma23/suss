import warnings

warnings.filterwarnings("ignore")

# Standard library
import asyncio
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

# Third party
from rich.rule import Rule
from rich.table import Table
from rich.syntax import Syntax
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, TextColumn, BarColumn

# Local
try:
    from bugnet.index import Index, File
    from bugnet.agent import Agent
except ImportError:
    from index import Index, File
    from agent import Agent


#########
# HELPERS
#########


@dataclass
class Config:
    file: Optional[str] = None  # Specific file to scan for bugs
    root_dir: Optional[str] = None
    max_iters: int = 5
    model: str = "openai/gpt-4o"


def is_model_available(model: str) -> bool:
    # TODO: Check that model is supported and that API keys are available
    return True


def clean_root_dir(root_dir: Optional[str] = None) -> Path:
    if root_dir is None:
        return Path.cwd()

    return Path(root_dir)


def get_config() -> Optional[Config]:
    parser = argparse.ArgumentParser(
        description=("Use AI to catch bugs in your code changes.")
    )
    parser.add_argument(
        "--file",
        type=str,
        required=False,
        default=None,
        help="A specific file to scan for bugs.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=False,
        default=None,
        help="Root directory of your codebase.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Get better results (slower, more LLM calls) by adding a refinement loop.",
    )
    parser.add_argument(
        "--max-iters",
        required=False,
        type=int,
        default=5,
        help="Max. # of iterations to run the agent per file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="The LLM model to use.",
    )
    args = parser.parse_args()

    if not is_model_available(args.model):
        print(f"API key for {args.model} not found. Please update your environment.")
        return None

    return Config(
        file=args.file,
        root_dir=clean_root_dir(args.root_dir),
        max_iters=args.max_iters,
        model=args.model,
    )


def get_changed_files(index: Index) -> List[File]:
    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines()

        changed_files = set()
        for line in status_output:
            if not line.strip():
                continue

            status = line[:2].strip()
            file_path = line[3:].strip()

            if status == "D":  # Skip deleted files
                continue

            changed_files.add(file_path)

        is_changed_file = lambda f: any(
            c_file.strip().lower() in f.rel_path.strip().lower()
            for c_file in changed_files
        )
        return [f for f in index.files if is_changed_file(f)]
    except subprocess.SubprocessError as e:
        return []


async def run_agent(agent: Agent, file: File, progress: Progress):
    task_id = progress.add_task(file.rel_path, total=agent.max_iters + 1, message="")

    def update_progress(message: str):
        progress.advance(task_id)
        progress.update(task_id, message=message)

    output = await agent.run(file, update_progress=update_progress)
    progress.update(task_id, completed=agent.max_iters + 1)
    return output


def print_bug_report(files: List[File], bug_sets: List[object], padding: int = 3):
    console = Console()
    console.print("[bold]Bug Report:[/bold]")
    for file, bugs in zip(files, bug_sets):
        for bug in bugs:
            file_path, confidence = file.rel_path, bug.confidence
            bug_description = bug.description
            start, end = bug.start, bug.end
            padded_start = max(1, start - padding)
            padded_end = min(end + padding, len(file.lines))
            code = "\n".join(file.lines[padded_start - 1 : padded_end])

            # Divider
            console.print(Rule(style="cyan"))

            # Title
            table = Table.grid(expand=True)
            table.add_column(justify="left")
            table.add_column(justify="right")
            lines = f"L{start}-{end}" if start != end else f"L{start}"
            left = f"[bold][cyan]{file_path}[/cyan][/bold] [cyan]({lines})[/cyan]"
            right = f"{confidence}%"
            if confidence <= 33:
                right = f"[red]{confidence}%[/red]"
            elif confidence <= 66:
                right = f"[yellow]{confidence}%[/yellow]"
            else:
                right = f"[green]{confidence}%[/green]"
            right += " confidence"
            table.add_row(left, right)
            console.print(table)
            console.print()

            # Code block
            console.print(
                Syntax(
                    code,
                    file.language,
                    theme="monokai",
                    line_numbers=True,
                    start_line=padded_start,
                    highlight_lines=list(range(start, end + 1)),
                )
            )
            console.print()

            # Bug description
            console.print(
                Markdown(
                    bug_description,
                    code_theme="monokai",
                    inline_code_lexer="python",
                    inline_code_theme="monokai",
                )
            )
            console.print()


######
# MAIN
######


async def main():
    config = get_config()
    index = Index(config.root_dir)
    agent = Agent(index, config.model, config.max_iters)
    files_to_analyze = get_changed_files(index)

    if not files_to_analyze:
        print("No changes detected")  # TODO: Improve
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True,
    ) as progress:
        tasks = []
        for file in files_to_analyze:
            task = run_agent(agent, file, progress)
            tasks.append(task)

        bug_sets = await asyncio.gather(*tasks)

    print_bug_report(files_to_analyze, bug_sets)
