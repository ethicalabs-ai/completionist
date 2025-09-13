#!/bin/bash

set -e

uv_command="uv"
default_args=("run" "python3" "-m" "completionist")

# Check if the first argument looks like a flag or if it's not a known command.
# If it's either, we'll assume the user wants to run the default command.
if [ "${1#-}" != "${1}" ] || [ -z "$(command -v "${1}")" ]; then
    set -- "$uv_command" "${default_args[@]}" "$@"
fi

# Replace the current shell with the final command.
exec "$@"