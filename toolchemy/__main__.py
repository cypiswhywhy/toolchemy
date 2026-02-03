import click

from toolchemy.ai.prompter import run_studio


@click.group(context_settings=dict(allow_interspersed_args=False))
def cli():
    pass


@cli.command(name="prompt-studio")
@click.option("--registry-dir", "-r", type=str, required=False, default=None, help="Optional path to the registry dir")
def prompt_studio(registry_dir: str | None = None):
    run_studio(registry_path=registry_dir)


if __name__ == '__main__':
    cli()
