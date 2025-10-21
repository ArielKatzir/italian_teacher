#!/usr/bin/env python3
"""
Student CLI - Command-line interface for students to view their homework.

Usage:
    python -m src.cli.student_cli homework list --student-id 1
    python -m src.cli.student_cli homework view --student-id 1 --homework-id 1
"""

import asyncio
import os
import sys

import httpx
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Student CLI - View homework assignments")
homework_app = typer.Typer(help="Homework management commands")

app.add_typer(homework_app, name="homework")

console = Console()

# API Base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ========== Homework Commands ==========


@homework_app.command("list")
def list_homework(
    student_id: int = typer.Option(..., "--student-id", "-s", help="Student ID"),
    status: str = typer.Option(
        "all",
        "--status",
        help="Filter by status (all, available, pending, in_progress, completed)",
    ),
):
    """List all homework for a student."""
    asyncio.run(_list_homework(student_id, status))


async def _list_homework(student_id: int, status: str):
    """List all homework for a student via API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/api/student/{student_id}/homework",
                params={"status": status},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            homework_list = data["homework"]

            if not homework_list:
                rprint(f"[yellow]No {status} homework found for student {student_id}.[/yellow]")
                return

            table = Table(
                title=f"Homework for Student {student_id} (Status: {status})",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Homework ID", style="cyan", width=12)
            table.add_column("Assignment ID", style="blue", width=14)
            table.add_column("# Exercises", style="green", width=12)
            table.add_column("Status", style="yellow", width=12)
            table.add_column("Created At", style="white")

            for hw in homework_list:
                num_exercises = len(hw.get("exercises", []))
                status_color = {
                    "pending": "yellow",
                    "available": "green",
                    "in_progress": "blue",
                    "completed": "cyan",
                }.get(hw["status"], "white")

                table.add_row(
                    str(hw["id"]),
                    str(hw["assignment_id"]),
                    str(num_exercises),
                    f"[{status_color}]{hw['status']}[/{status_color}]",
                    hw["created_at"][:19],
                )

            console.print(table)
            rprint(f"\n[bold]Total homework:[/bold] {data['total']}")
            rprint(
                "\n[dim]Use 'homework view --student-id <id> --homework-id <id>' to see exercises[/dim]"
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                rprint(f"[red]✗ Student {student_id} not found.[/red]")
            else:
                rprint(f"[red]✗ Error: {e.response.json().get('detail', str(e))}[/red]")
            sys.exit(1)
        except Exception as e:
            rprint(f"[red]✗ Connection error: {e}[/red]")
            rprint("[yellow]Make sure the API is running (./run_api.sh)[/yellow]")
            sys.exit(1)


@homework_app.command("view")
def view_homework(
    student_id: int = typer.Option(..., "--student-id", "-s", help="Student ID"),
    homework_id: int = typer.Option(..., "--homework-id", "-h", help="Homework ID"),
):
    """View a specific homework with all exercises."""
    asyncio.run(_view_homework(student_id, homework_id))


async def _view_homework(student_id: int, homework_id: int):
    """View a specific homework via API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/api/student/{student_id}/homework/{homework_id}",
                timeout=10.0,
            )
            response.raise_for_status()
            homework = response.json()

            # Header
            status_emoji = {
                "pending": "⏳",
                "available": "✅",
                "in_progress": "📝",
                "completed": "🎉",
            }.get(homework["status"], "📋")

            rprint(f"\n{status_emoji} [bold]Homework #{homework['id']}[/bold]")
            rprint(f"[bold]Assignment ID:[/bold] {homework['assignment_id']}")
            rprint(f"[bold]Status:[/bold] {homework['status']}")
            rprint(f"[bold]Created:[/bold] {homework['created_at'][:19]}")

            exercises = homework.get("exercises", [])

            if not exercises:
                rprint("\n[yellow]⏳ No exercises generated yet. Please check back later.[/yellow]")
                return

            rprint(f"\n[bold]Total Exercises:[/bold] {len(exercises)}\n")

            # Display exercises
            for i, exercise in enumerate(exercises, 1):
                exercise_type = exercise["type"]
                question = exercise["question"]
                correct_answer = exercise["correct_answer"]
                explanation = exercise.get("explanation", "")
                options = exercise.get("options", [])

                # Exercise type styling
                type_style = {
                    "fill_in_blank": "blue",
                    "translation": "green",
                    "multiple_choice": "magenta",
                }.get(exercise_type, "white")

                # Create exercise panel
                content = f"[bold]Question:[/bold] {question}\n"

                if options:
                    content += "\n[bold]Options:[/bold]\n"
                    for j, option in enumerate(options, 1):
                        content += f"  {j}. {option}\n"

                content += f"\n[bold]Answer:[/bold] {correct_answer}"

                if explanation:
                    content += f"\n\n[dim]{explanation}[/dim]"

                panel = Panel(
                    content,
                    title=f"[{type_style}]Exercise {i}: {exercise_type.replace('_', ' ').title()}[/{type_style}]",
                    border_style=type_style,
                    expand=False,
                )

                console.print(panel)
                print()  # Add spacing

            rprint("[green]Good luck with your homework! 🇮🇹[/green]")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                rprint(f"[red]✗ Homework {homework_id} not found for student {student_id}.[/red]")
            else:
                rprint(f"[red]✗ Error: {e.response.json().get('detail', str(e))}[/red]")
            sys.exit(1)
        except Exception as e:
            rprint(f"[red]✗ Error: {e}[/red]")
            sys.exit(1)


if __name__ == "__main__":
    app()
