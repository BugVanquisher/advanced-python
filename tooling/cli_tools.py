"""CLI Tools with Click/Typer"""

import click
import typer
from typing import Optional, List
from pathlib import Path
import json
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import track
import time


# Click example
@click.group()
@click.version_option(version='1.0.0')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """A sample CLI application using Click."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    if verbose:
        click.echo('Verbose mode enabled')


@cli.command()
@click.argument('name')
@click.option('--count', '-c', default=1, help='Number of greetings')
@click.option('--greeting', '-g', default='Hello', help='Greeting to use')
@click.pass_context
def greet(ctx, name, count, greeting):
    """Greet someone multiple times."""
    for i in range(count):
        click.echo(f'{greeting}, {name}!')

    if ctx.obj['verbose']:
        click.echo(f'Greeted {name} {count} times')


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']),
              default='json', help='Output format')
def convert(input_file, output_file, output_format):
    """Convert files between formats."""
    click.echo(f'Converting {input_file} to {output_file} as {output_format}')

    # Simulate conversion
    data = {'converted': True, 'source': input_file, 'format': output_format}

    with open(output_file, 'w') as f:
        if output_format == 'json':
            json.dump(data, f, indent=2)
        else:
            f.write('converted,source,format\n')
            f.write(f"True,{input_file},{output_format}\n")

    click.echo(f'✅ Conversion complete!')


@cli.command()
@click.option('--url', prompt='Service URL', help='URL of the service to check')
@click.option('--timeout', default=5, help='Timeout in seconds')
def health_check(url, timeout):
    """Check the health of a service."""
    with click.progressbar(range(timeout), label='Checking service') as bar:
        for i in bar:
            time.sleep(1)

    click.echo(f'✅ Service at {url} is healthy')


# Typer example
app = typer.Typer(help="A sample CLI application using Typer")
console = Console()


@app.command()
def hello(
    name: str = typer.Argument(..., help="Name to greet"),
    count: int = typer.Option(1, "--count", "-c", help="Number of greetings"),
    formal: bool = typer.Option(False, "--formal", help="Use formal greeting")
):
    """Greet someone with Typer."""
    greeting = "Good day" if formal else "Hello"

    for i in range(count):
        console.print(f"[bold green]{greeting}[/bold green], [blue]{name}[/blue]!")


@app.command()
def process_files(
    files: List[Path] = typer.Argument(..., help="Files to process"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done")
):
    """Process multiple files."""
    if output_dir and not output_dir.exists():
        if not dry_run:
            output_dir.mkdir(parents=True)
        console.print(f"[yellow]Would create directory: {output_dir}[/yellow]")

    table = Table(title="File Processing")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size", justify="right")

    for file_path in track(files, description="Processing files..."):
        if file_path.exists():
            size = file_path.stat().st_size
            status = "Would process" if dry_run else "Processed"
            table.add_row(str(file_path), status, f"{size} bytes")

            if not dry_run:
                time.sleep(0.1)  # Simulate processing
        else:
            table.add_row(str(file_path), "[red]Not found[/red]", "")

    console.print(table)


@app.command()
def config(
    set_key: Optional[str] = typer.Option(None, "--set", help="Set config key=value"),
    get_key: Optional[str] = typer.Option(None, "--get", help="Get config value"),
    list_all: bool = typer.Option(False, "--list", help="List all config")
):
    """Manage configuration."""
    config_file = Path("config.json")

    # Load existing config
    if config_file.exists():
        with open(config_file) as f:
            config_data = json.load(f)
    else:
        config_data = {}

    if set_key:
        try:
            key, value = set_key.split('=', 1)
            config_data[key] = value

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            console.print(f"[green]Set {key} = {value}[/green]")
        except ValueError:
            console.print("[red]Error: Use format key=value[/red]")
            raise typer.Exit(1)

    elif get_key:
        value = config_data.get(get_key)
        if value is not None:
            console.print(f"{get_key} = {value}")
        else:
            console.print(f"[yellow]Key '{get_key}' not found[/yellow]")

    elif list_all:
        if config_data:
            table = Table(title="Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")

            for key, value in config_data.items():
                table.add_row(key, str(value))

            console.print(table)
        else:
            console.print("[yellow]No configuration found[/yellow]")
    else:
        console.print("[yellow]Use --set, --get, or --list[/yellow]")


@app.command()
def interactive():
    """Interactive mode with prompts."""
    console.print("[bold blue]Welcome to interactive mode![/bold blue]")

    name = typer.prompt("What's your name?")
    age = typer.prompt("What's your age?", type=int)

    is_developer = typer.confirm("Are you a developer?")

    if is_developer:
        language = typer.prompt(
            "What's your favorite programming language?",
            default="Python"
        )
        console.print(f"[green]Nice! {language} is a great choice![/green]")

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"Name: {name}")
    console.print(f"Age: {age}")
    console.print(f"Developer: {'Yes' if is_developer else 'No'}")


def demonstrate_cli_tools():
    """Demonstrate CLI tools."""
    console.print("[bold blue]=== CLI Tools Demo ===[/bold blue]")

    console.print("\n[yellow]Click CLI Example:[/yellow]")
    console.print("Run: python cli_tools.py greet Alice --count 3")
    console.print("Run: python cli_tools.py health-check --url http://example.com")

    console.print("\n[yellow]Typer CLI Example:[/yellow]")
    console.print("Run: python -c 'from cli_tools import app; app()' hello Alice --count 2")
    console.print("Run: python -c 'from cli_tools import app; app()' config --set debug=true")

    # Demonstrate programmatic usage
    console.print("\n[yellow]Programmatic Demo:[/yellow]")

    # Create a demo file
    demo_file = Path("demo.txt")
    demo_file.write_text("Hello, World!")

    try:
        # Simulate Typer command
        console.print("Processing demo file...")
        if demo_file.exists():
            size = demo_file.stat().st_size
            console.print(f"File: {demo_file}, Size: {size} bytes")
    finally:
        # Clean up
        if demo_file.exists():
            demo_file.unlink()

        config_file = Path("config.json")
        if config_file.exists():
            config_file.unlink()


if __name__ == "__main__":
    # Run Click CLI if called directly
    if len(sys.argv) > 1 and sys.argv[1] in ['greet', 'convert', 'health-check']:
        cli()
    else:
        demonstrate_cli_tools()