import warnings

warnings.filterwarnings("ignore")

# Standard library
import asyncio
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass

# Third party
from rich.rule import Rule
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.console import Console, Group
from rich.progress import Progress, TextColumn, BarColumn

# Local
try:
    from suss.agent import Agent
    from suss.index import Index, File
    from suss.constants import MAX_TOOL_CALLS
except ImportError:
    from agent import Agent
    from index import Index, File
    from constants import MAX_TOOL_CALLS


#########
# HELPERS
#########


@dataclass
class Config:
    file: str | None = None  # Specific file to scan for bugs
    root_dir: str | None = None
    max_iters: int = MAX_TOOL_CALLS
    model: str = "openai/gpt-4o"


def clean_root_dir(root_dir: str | None = None) -> Path:
    if root_dir is None:
        return Path.cwd()

    return Path(root_dir)


def get_config() -> Config | None:
    parser = argparse.ArgumentParser(
        description=("Use AI to catch bugs in your code changes.")
    )
    parser.add_argument(
        "--file",
        type=str,
        required=False,
        default=None,
        help="Relative path to a specific file to scan for bugs.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=False,
        default=None,
        help="Root directory of your codebase.",
    )
    parser.add_argument(
        "--max-iters",
        required=False,
        type=int,
        default=MAX_TOOL_CALLS,
        help="Max. # of iterations to run the agent per file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="The LLM model to use.",
    )
    args = parser.parse_args()

    return Config(
        file=args.file,
        root_dir=clean_root_dir(args.root_dir),
        max_iters=args.max_iters,
        model=args.model,
    )


def get_changed_files(index: Index, config: Config) -> list[File]:
    try:
        if config.file:
            return [f for f in index.files if f.rel_path.lower() == config.file.lower()]

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


async def run_agent(agent: Agent, file: File, progress: Progress, update_log_panel):
    task_id = progress.add_task(file.rel_path, total=agent.max_iters + 1, message="")

    def update_progress(message: str):
        progress.advance(task_id)
        progress.update(task_id, message=message)
        update_log_panel(message, file.rel_path)

    output = await agent.run(file, update_progress=update_progress)
    progress.update(task_id, completed=agent.max_iters + 1)
    return output


def print_bug_report(files: list[File], bug_sets: list[object], padding: int = 3):
    console = Console()
    # console.print("[bold]Bug Report:[/bold]")
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


def main():
    async def async_main():
        console = Console()
        config = get_config()
        index = Index(config.root_dir)
        agent = Agent(index, config.model, config.max_iters)
        files_to_analyze = get_changed_files(index, config)

        if not files_to_analyze:
            if not config.file:
                console.print("No code changes detected. Aborting analysis.")
            else:
                console.print(f"No such file: {config.file}. Aborting analysis.")

            return

        console.print("[bold]Analyzing files...[/bold]")
        console.print("")

        log_panel = Text("")
        file_logs = []

        progress = Progress(
            TextColumn("[progress.description]{task.description}", justify="left"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            expand=True,
        )

        display_group = Group(progress, Text(""), Rule(style="cyan"), log_panel)

        with Live(display_group, refresh_per_second=10, transient=False) as live:

            def update_log_panel(message: str, file_path: str):
                nonlocal file_logs
                file_logs.append(f"[cyan]{file_path}:[/cyan] {message}")
                if len(file_logs) > 5:
                    file_logs.pop(0)
                display_group.renderables[-1] = Text.from_markup("\n".join(file_logs))

            tasks = []
            for file in files_to_analyze:
                task = run_agent(agent, file, progress, update_log_panel)
                tasks.append(task)

            bug_sets = await asyncio.gather(*tasks)

        console.clear()
        print_bug_report(files_to_analyze, bug_sets)

    asyncio.run(async_main())
