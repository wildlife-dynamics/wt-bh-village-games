#!/bin/bash

python_version=$1

command="pixi run \
--manifest-path src/ecoscope-workflows-ext-bh-village-games/pyproject.toml \
--environment test-py${python_version} \
pytest src/ecoscope-workflows-ext-bh-village-games/tests -vv"

shift 1
if [ -n "$*" ]; then
    extra_args=$*
    command="$command $extra_args"
fi

echo "Running command: $command"
eval $command
