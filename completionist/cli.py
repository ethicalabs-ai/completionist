import click

# from completionist.commands.build import build_cmd
from completionist.commands.complete import complete_cmd
# from completionist.commands.compose import compose_cmd
# from completionist.commands.translate import translate_cmd


@click.group()
def entry_point():
    pass


# entry_point.add_command(build_cmd)
entry_point.add_command(complete_cmd)
# entry_point.add_command(compose_cmd)
# entry_point.add_command(translate_cmd)
