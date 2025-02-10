# Standard library
import re
from pathlib import Path
from typing import Generator, List, Optional
from collections import namedtuple
from functools import cached_property

# Third party
from sortedcollections import OrderedSet
from tree_sitter_languages import get_parser

# Local
try:
    from bugnet.resources import BOILERPLATE_REGEXES, LANGUAGE_EXTENSIONS
except ImportError:
    from resources import BOILERPLATE_REGEXES, LANGUAGE_EXTENSIONS


MAX_HEADER_SIZE = 10
MARGIN = 3
PADDING = 1
MIN_BLOCK_SIZE = 5
SHOW_FACTOR = 0.10
MAX_TO_SHOW = 25
MIN_TO_SHOW = 5


#########
# HELPERS
#########


HeaderPart = namedtuple("HeaderPart", ["size", "start", "end"])
LineRange = namedtuple("LineRange", ["start", "end"])


class LineContext(object):
    def __init__(self, line_num: int):
        self.line_num = line_num
        self.scopes = OrderedSet()
        self.nodes = []
        self.header_parts = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_header_part(self, start: int, end: int):
        size = end - start
        if not size:
            return

        self.header_parts[start].append(HeaderPart(size, start, end))

    def add_scope(self, start: int):
        self.scopes.add(start)

    @cached_property
    def header(self) -> LineRange:
        header_parts = sorted(self.header_parts)
        if len(header_parts) > 1:
            size, start, end = self.header_parts[0]
            if size > MAX_HEADER_SIZE:
                end = start + MAX_HEADER_SIZE
        else:
            start = self.line_num
            end = self.line_num + 1

        return LineRange(start, end)


def should_index_file(file: Path) -> bool:
    # TODO: Use .gitignore

    if file.is_dotfile:
        return False

    if not file.extension:
        return False

    if not file.is_code_file:
        return False

    if file.is_boilerplate:
        return False

    return True


def should_index_dir(dir: Path) -> bool:
    # TODO: Use .gitignore
    return not any(re.search(pattern, dir) for pattern in BOILERPLATE_REGEXES)


def enumerate_files(root_dir: Path, sub_dir: Optional[Path] = None):
    # TODO: Parallelize
    sub_dir = root_dir if not sub_dir else sub_dir

    for item in sub_dir.iterdir():
        if item.is_dir():
            if not should_index_dir(item):
                continue

            yield from enumerate_files(root_dir, item)
        elif item.is_file():
            if not should_index_file(item):
                continue

            yield File(root_dir, item)


######
# MAIN
######


class File(object):
    def __init__(self, root_dir: Path, path: Path):
        self.root_dir = root_dir
        self.path = path

        with open(self.path, "r") as f:
            self.code = f.read()

        self.lines = self.code.splitlines()
        self.num_lines = len(self.lines) + 1
        self.line_contexts = [LineContext(i) for i in range(self.num_lines)]

    def __hash__(self) -> str:
        return hash(self.abs_path)

    @cached_property
    def last_lineno(self) -> int:
        return len(self.lines)

    @cached_property
    def content(self) -> str:
        content = ""
        for line_num, line in enumerate(self.lines):
            content += f"{line_num + 1} {line}\n"

        return content

    @cached_property
    def rel_path(self) -> str:
        return str(self.path.relative_to(self.root_dir))

    @cached_property
    def abs_path(self) -> str:
        return str(self.path)

    @cached_property
    def name(self) -> str:
        return self.path.name

    @cached_property
    def extension(self) -> str:
        return self.path.suffix

    @cached_property
    def is_dotfile(self) -> bool:
        return self.name.startswith(".")

    @cached_property
    def is_code_file(self) -> bool:
        if self.extension not in LANGUAGE_EXTENSIONS:
            return False

        if LANGUAGE_EXTENSIONS[self.extension]["is_code"]:
            return True

        return False

    @cached_property
    def is_boilerplate(self) -> bool:
        return any(re.search(pattern, self.rel_path) for pattern in BOILERPLATE_REGEXES)

    def index(self):
        def recurse_tree(node):
            start, end = node.start_point, node.end_point
            start_line, end_line = start[0], end[0]

            for line in range(start_line, end_line + 1):
                line_context = self.line_contexts[line]

                if line == start_line:
                    line_context.add_node(node)
                    line_context.add_header_part(start_line, end_line)

                line_context.add_scope(start_line)

            for child in node.children:
                recurse_tree(child)

        parser = get_parser(self.language)
        tree = parser.parse(bytes(self.code, "utf8"))
        recurse_tree(tree.root_node)

    # Adapted from Paul Gauthier's grep-ast
    def search(self, pattern: str):
        parent_scopes = OrderedSet()
        matches = OrderedSet()
        show_lines = OrderedSet()

        for line_num, line in enumerate(self.lines):
            if re.search(pattern, line, re.IGNORECASE):  # 0 for no ignore
                matches.add(line_num)
                show_lines.add(line_num)

        if not matches:
            return []

        # Add padding
        for line in show_lines:
            for new_line in range(line - PADDING, line + PADDING + 1):
                if new_line >= self.num_lines:
                    continue
                if new_line < 0:
                    continue

                show_lines.add(new_line)

        def get_last_line_of_scope(i: int) -> int:
            line_context = self.line_contexts[i]
            last_line = max(node.end_point[0] for node in line_context.nodes)
            return last_line

        def get_all_children(node) -> list:
            children = [node]
            for child in node.children:
                children += get_all_children(child)

            return children

        def add_parent_scopes(i: int):
            if i in parent_scopes:
                return

            parent_scopes.add(i)

            if i >= len(self.line_contexts):
                return

            for scope in self.line_contexts[i].scopes:
                header = self.line_contexts[scope].header
                show_lines.update(range(header.start, header.end))
                last_line = get_last_line_of_scope(scope)
                add_parent_scopes(last_line)

        def add_child_context(i: int):
            if not self.line_contexts[i].nodes:
                return

            last_line = get_last_line_of_scope(i)
            size = last_line - i
            if size < MIN_BLOCK_SIZE:
                show_lines.update(range(i, last_line + 1))
                return

            children = []
            for node in self.line_contexts[i].nodes:
                children += get_all_children(node)

            children = sorted(
                children,
                key=lambda node: node.end_point[0] - node.start_point[0],
                reverse=True,
            )
            num_show_lines = len(show_lines)
            max_to_show = max(min(size * SHOW_FACTOR, MAX_TO_SHOW), MIN_TO_SHOW)

            for child in children:
                if len(show_lines) > num_show_lines + max_to_show:
                    break

                child_start = child.start_point[0]
                add_parent_scopes(child_start)

        # Add the bottom line + parent context
        bottom_line = self.num_lines - 2
        show_lines.add(bottom_line)
        add_parent_scopes(bottom_line)

        # Add parent context for matched lines
        for line_num in matches:
            add_parent_scopes(line_num)

        # Then add child context
        for line_num in matches:
            add_child_context(line_num)

        # Add the top margin lines of the file
        show_lines.update(range(MARGIN))

        # Close small gaps
        show_lines = OrderedSet(sorted(show_lines))
        closed_show_lines = OrderedSet(show_lines)
        for line_num in range(len(show_lines) - 1):
            if show_lines[line_num + 1] - show_lines[line_num] == 2:
                closed_show_lines.add(show_lines[line_num] + 1)
        for line_num, line in enumerate(self.lines):
            if line_num not in closed_show_lines:
                continue

            if (
                self.lines[line_num].strip()
                and line_num < self.num_lines - 2
                and not self.lines[line_num + 1].strip()
            ):
                closed_show_lines.add(line_num + 1)
        show_lines = OrderedSet(sorted(closed_show_lines))


class Chunk(object):
    def __init__(self, line_nums: List[int], file: File):
        """
        Line #s are 1-indexed TODO: ?
        Chunk is not necessarily contiguous.
        """

        self.line_nums = OrderedSet(line_nums)
        self.file = file

    def __hash__(self) -> str:
        line_nums_str = ",".join(self.line_nums)
        file_str = self.file.abs_path
        return hash(f"{file_str}::{line_nums_str}")

    @cached_property
    def lines(self) -> List[str]:
        return [self.file.lines[i] for i in self.line_nums]

    def to_string(self, line_nums: bool = True, dots: bool = True) -> str:
        output_str = ""

        use_dots = not (0 in self.line_nums)
        for line_num, line in enumerate(self.file.lines):
            if line_num not in self.line_nums:
                if use_dots and dots:
                    output_str += "⋮...\n"
                    use_dots = False

                continue

            output_str += f"{line_num + 1} {line}\n" if line_nums else f"{line}\n"
            use_dots = True

        return output_str


class Index(object):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.files = {f.rel_path: f for f in enumerate_files(self.root_dir)}

    def get_file(self, rel_path: str) -> File:
        return self.files[rel_path]

    def search_code(self, pattern: str) -> List[Chunk]:
        # TODO: Parallelize

        results = OrderedSet()
        for file in self.files.values():
            line_matches = file.search(pattern)
            if not line_matches:
                continue

            chunk = Chunk(line_matches, file)
            results.add(chunk)

        return list(results)

    def search_files(self, pattern: str) -> List[File]:
        pass
