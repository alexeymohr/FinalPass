from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import click

T = TypeVar("T")


class WizardBack(Exception):
    """Raised when the user chooses to step back one menu."""


class WizardQuit(Exception):
    """Raised when the user chooses to exit the wizard."""


@dataclass(frozen=True)
class Choice(Generic[T]):
    label: str
    value: T


def choose_one(
    title: str,
    options: list[Choice[T]],
    *,
    default_index: int | None = None,
    allow_back: bool = True,
    auto_select: bool = True,
    auto_select_notice: str | None = None,
) -> T:
    if not options:
        raise ValueError("choose_one requires at least one option")

    if auto_select and len(options) == 1:
        if auto_select_notice:
            click.echo(auto_select_notice)
        return options[0].value

    while True:
        click.echo(title)
        for index, option in enumerate(options, start=1):
            click.echo(f"{index}. {option.label}")
        if allow_back:
            click.echo("0. Back")
        click.echo("q. Quit")

        raw = _prompt(default=str(default_index) if default_index is not None else None)
        if raw.lower() == "q":
            raise WizardQuit
        if allow_back and raw == "0":
            raise WizardBack
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1].value

        click.echo(_invalid_menu_message(len(options), allow_back=allow_back))


def choose_many(
    title: str,
    options: list[Choice[T]],
    *,
    allow_back: bool = True,
) -> list[T]:
    if not options:
        raise ValueError("choose_many requires at least one option")

    while True:
        click.echo(title)
        for index, option in enumerate(options, start=1):
            click.echo(f"{index}. {option.label}")
        if allow_back:
            click.echo("0. Back")
        click.echo("q. Quit")

        raw = _prompt()
        if raw.lower() == "q":
            raise WizardQuit
        if allow_back and raw == "0":
            raise WizardBack

        indexes = _parse_index_list(raw)
        if indexes is None:
            click.echo("Enter one or more numbers separated by commas.")
            continue
        if not indexes:
            click.echo("Choose at least one option.")
            continue
        if any(index < 1 or index > len(options) for index in indexes):
            click.echo(_invalid_menu_message(len(options), allow_back=allow_back))
            continue

        return [options[index - 1].value for index in indexes]


def prompt_text(
    title: str,
    *,
    default: str | None = None,
    allow_back: bool = True,
    allow_blank: bool = False,
) -> str:
    while True:
        click.echo(title)
        if allow_back:
            click.echo("0. Back")
        click.echo("q. Quit")

        raw = _prompt(default=default)
        if raw.lower() == "q":
            raise WizardQuit
        if allow_back and raw == "0":
            raise WizardBack
        if raw or allow_blank:
            return raw
        click.echo("Enter a value or choose q to quit.")


def print_section(title: str) -> None:
    click.echo()
    click.echo(title)


def _prompt(*, default: str | None = None) -> str:
    raw = input("> ").strip()
    if raw == "" and default is not None:
        return default
    return raw


def _invalid_menu_message(count: int, *, allow_back: bool) -> str:
    if allow_back:
        return f"Invalid choice. Enter 1-{count}, 0 for Back, or q to quit."
    return f"Invalid choice. Enter 1-{count} or q to quit."


def _parse_index_list(raw: str) -> list[int] | None:
    if not raw:
        return []
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        return None
    if any(not part.isdigit() for part in parts):
        return None

    seen: set[int] = set()
    out: list[int] = []
    for part in parts:
        index = int(part)
        if index in seen:
            continue
        seen.add(index)
        out.append(index)
    return out
