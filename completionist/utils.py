import sys
import click


def handle_error(message):
    """Prints an error message and exits the program."""
    click.echo(click.style(f"Error: {message}", fg="red"), err=True)
    sys.exit(1)


def read_file_content(file_path):
    """Helper function to read content from a file path."""
    if not file_path:
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        handle_error(f"Error reading file {file_path}: {e}")
