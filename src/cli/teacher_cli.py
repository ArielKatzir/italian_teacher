#!/usr/bin/env python3
"""
Teacher CLI - Command-line interface for teachers to manage students and assignments.

Usage:
    python -m src.cli.teacher_cli student create --name "Mario Rossi" --email "mario@example.com"
    python -m src.cli.teacher_cli student list
    python -m src.cli.teacher_cli student delete --id 1

    python -m src.cli.teacher_cli assignment create --student-ids 1,2 --level A2 --topic "daily routines"
    python -m src.cli.teacher_cli assignment list
    python -m src.cli.teacher_cli assignment status --id 1
"""

import asyncio
import os
import sys
from typing import Optional

import httpx
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Teacher CLI - Manage students and assignments")
student_app = typer.Typer(help="Student management commands")
assignment_app = typer.Typer(help="Assignment management commands")

app.add_typer(student_app, name="student")
app.add_typer(assignment_app, name="assignment")

console = Console()

# API Base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ========== Student Commands ==========


@student_app.command("create")
def create_student(
    name: str = typer.Option(..., "--name", "-n", help="Student name"),
    email: str = typer.Option(..., "--email", "-e", help="Student email"),
):
    """Create a new student."""
    asyncio.run(_create_student(name, email))


async def _create_student(name: str, email: str):
    """Create a new student via API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_BASE_URL}/api/teacher/students",
                json={"name": name, "email": email},
                timeout=10.0,
            )
            response.raise_for_status()
            student = response.json()

            rprint(f"[green]✓ Student created successfully![/green]")
            rprint(f"[bold]ID:[/bold] {student['id']}")
            rprint(f"[bold]Name:[/bold] {student['name']}")
            rprint(f"[bold]Email:[/bold] {student['email']}")

        except httpx.HTTPStatusError as e:
            rprint(f"[red]✗ Error: {e.response.json().get('detail', str(e))}[/red]")
            sys.exit(1)
        except Exception as e:
            rprint(f"[red]✗ Connection error: {e}[/red]")
            rprint("[yellow]Make sure the API is running (./run_api.sh)[/yellow]")
            sys.exit(1)


@student_app.command("list")
def list_students():
    """List all students."""
    asyncio.run(_list_students())


async def _list_students():
    """List all students via API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/api/teacher/students", timeout=10.0)
            response.raise_for_status()
            students = response.json()

            if not students:
                rprint("[yellow]No students found.[/yellow]")
                return

            table = Table(title="Students", show_header=True, header_style="bold magenta")
            table.add_column("ID", style="cyan", width=6)
            table.add_column("Name", style="green")
            table.add_column("Email", style="blue")
            table.add_column("Created At", style="white")

            for student in students:
                table.add_row(
                    str(student["id"]),
                    student["name"],
                    student["email"],
                    student["created_at"][:19],  # Remove microseconds
                )

            console.print(table)
            rprint(f"\n[bold]Total students:[/bold] {len(students)}")

        except Exception as e:
            rprint(f"[red]✗ Error: {e}[/red]")
            sys.exit(1)


@student_app.command("delete")
def delete_student(
    student_id: int = typer.Option(..., "--id", "-i", help="Student ID to delete"),
):
    """Delete a student and all their homework."""
    if not typer.confirm(f"Are you sure you want to delete student ID {student_id}?"):
        rprint("[yellow]Cancelled.[/yellow]")
        return

    asyncio.run(_delete_student(student_id))


async def _delete_student(student_id: int):
    """Delete a student via API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(
                f"{API_BASE_URL}/api/teacher/students/{student_id}", timeout=10.0
            )
            response.raise_for_status()

            rprint(f"[green]✓ Student {student_id} deleted successfully![/green]")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                rprint(f"[red]✗ Student {student_id} not found.[/red]")
            else:
                rprint(f"[red]✗ Error: {e.response.json().get('detail', str(e))}[/red]")
            sys.exit(1)
        except Exception as e:
            rprint(f"[red]✗ Error: {e}[/red]")
            sys.exit(1)


# ========== Assignment Commands ==========


@assignment_app.command("create")
def create_assignment(
    student_ids: str = typer.Option(..., "--student-ids", "-s", help="Comma-separated student IDs"),
    level: str = typer.Option(..., "--level", "-l", help="CEFR level (A1, A2, B1, B2, C1, C2)"),
    topic: Optional[str] = typer.Option(
        None, "--topic", "-t", help="Topic (e.g., 'daily routines')"
    ),
    grammar_focus: Optional[str] = typer.Option(
        None, "--grammar", "-g", help="Grammar focus (e.g., 'present_tense')"
    ),
    quantity: int = typer.Option(5, "--quantity", "-q", help="Number of exercises"),
    exercise_types: str = typer.Option(
        "fill_in_blank,translation,multiple_choice",
        "--types",
        help="Comma-separated exercise types",
    ),
):
    """Create a new homework assignment."""
    asyncio.run(
        _create_assignment(student_ids, level, topic, grammar_focus, quantity, exercise_types)
    )


async def _create_assignment(
    student_ids: str,
    level: str,
    topic: Optional[str],
    grammar_focus: Optional[str],
    quantity: int,
    exercise_types: str,
):
    """Create a new assignment via API."""
    # Parse student IDs
    try:
        student_id_list = [int(sid.strip()) for sid in student_ids.split(",")]
    except ValueError:
        rprint("[red]✗ Invalid student IDs. Use comma-separated integers (e.g., '1,2,3')[/red]")
        sys.exit(1)

    # Parse exercise types
    exercise_type_list = [t.strip() for t in exercise_types.split(",")]

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_BASE_URL}/api/teacher/assignments",
                json={
                    "student_ids": student_id_list,
                    "cefr_level": level,
                    "topic": topic,
                    "grammar_focus": grammar_focus,
                    "quantity": quantity,
                    "exercise_types": exercise_type_list,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            assignment = response.json()

            rprint(f"[green]✓ Assignment created successfully![/green]")
            rprint(f"[bold]Assignment ID:[/bold] {assignment['id']}")
            rprint(f"[bold]Status:[/bold] {assignment['status']}")
            rprint(f"[bold]CEFR Level:[/bold] {assignment['cefr_level']}")
            rprint(f"[bold]Topic:[/bold] {assignment['topic'] or 'N/A'}")
            rprint(f"[bold]Grammar:[/bold] {assignment['grammar_focus'] or 'N/A'}")
            rprint(f"[bold]Quantity:[/bold] {assignment['quantity']}")
            rprint(f"[bold]Students:[/bold] {', '.join(map(str, assignment['student_ids']))}")
            rprint(
                "\n[yellow]⚙ Background generation started. Use 'assignment status' to check progress.[/yellow]"
            )

        except httpx.HTTPStatusError as e:
            rprint(f"[red]✗ Error: {e.response.json().get('detail', str(e))}[/red]")
            sys.exit(1)
        except Exception as e:
            rprint(f"[red]✗ Error: {e}[/red]")
            sys.exit(1)


@assignment_app.command("list")
def list_assignments():
    """List all assignments."""
    asyncio.run(_list_assignments())


async def _list_assignments():
    """List all assignments via API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/api/teacher/assignments", timeout=10.0)
            response.raise_for_status()
            assignments = response.json()

            if not assignments:
                rprint("[yellow]No assignments found.[/yellow]")
                return

            table = Table(title="Assignments", show_header=True, header_style="bold magenta")
            table.add_column("ID", style="cyan", width=6)
            table.add_column("Level", style="green", width=8)
            table.add_column("Topic", style="blue")
            table.add_column("Students", style="white", width=12)
            table.add_column("Status", style="yellow", width=12)
            table.add_column("Created At", style="white")

            for assignment in assignments:
                status_color = {
                    "pending": "yellow",
                    "generating": "blue",
                    "completed": "green",
                    "failed": "red",
                }.get(assignment["status"], "white")

                table.add_row(
                    str(assignment["id"]),
                    assignment["cefr_level"],
                    assignment["topic"] or "N/A",
                    ", ".join(map(str, assignment["student_ids"])),
                    f"[{status_color}]{assignment['status']}[/{status_color}]",
                    assignment["created_at"][:19],
                )

            console.print(table)
            rprint(f"\n[bold]Total assignments:[/bold] {len(assignments)}")

        except Exception as e:
            rprint(f"[red]✗ Error: {e}[/red]")
            sys.exit(1)


@assignment_app.command("status")
def get_assignment_status(
    assignment_id: int = typer.Option(..., "--id", "-i", help="Assignment ID"),
):
    """Check the status of an assignment."""
    asyncio.run(_get_assignment_status(assignment_id))


async def _get_assignment_status(assignment_id: int):
    """Get assignment status via API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/api/teacher/assignments/{assignment_id}", timeout=10.0
            )
            response.raise_for_status()
            assignment = response.json()

            # Status indicators
            status = assignment["status"]
            status_emoji = {
                "pending": "⏳",
                "generating": "⚙️",
                "completed": "✅",
                "failed": "❌",
            }.get(status, "❓")

            rprint(f"\n[bold]Assignment #{assignment['id']} Status[/bold]")
            rprint(f"{status_emoji} [bold]Status:[/bold] {status.upper()}")
            rprint(f"[bold]CEFR Level:[/bold] {assignment['cefr_level']}")
            rprint(f"[bold]Topic:[/bold] {assignment['topic'] or 'N/A'}")
            rprint(f"[bold]Grammar:[/bold] {assignment['grammar_focus'] or 'N/A'}")
            rprint(f"[bold]Quantity:[/bold] {assignment['quantity']}")
            rprint(f"[bold]Exercise Types:[/bold] {', '.join(assignment['exercise_types'])}")
            rprint(f"[bold]Students:[/bold] {', '.join(map(str, assignment['student_ids']))}")
            rprint(f"[bold]Created:[/bold] {assignment['created_at'][:19]}")

            if assignment.get("completed_at"):
                rprint(f"[bold]Completed:[/bold] {assignment['completed_at'][:19]}")

            if assignment.get("error_message"):
                rprint(f"[red][bold]Error:[/bold] {assignment['error_message']}[/red]")

            # Status-specific messages
            if status == "pending":
                rprint("\n[yellow]⏳ Assignment is queued for generation.[/yellow]")
            elif status == "generating":
                rprint(
                    "\n[blue]⚙️ Exercises are being generated using Colab GPU... Please wait.[/blue]"
                )
            elif status == "completed":
                rprint(
                    f"\n[green]✅ Assignment completed! Students can now view their homework.[/green]"
                )
            elif status == "failed":
                rprint("\n[red]❌ Assignment generation failed. Check error message above.[/red]")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                rprint(f"[red]✗ Assignment {assignment_id} not found.[/red]")
            else:
                rprint(f"[red]✗ Error: {e.response.json().get('detail', str(e))}[/red]")
            sys.exit(1)
        except Exception as e:
            rprint(f"[red]✗ Error: {e}[/red]")
            sys.exit(1)


if __name__ == "__main__":
    app()
