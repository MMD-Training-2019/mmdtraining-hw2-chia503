#!/bin/bash
# Shell script wrapper for Python

REPODIR=$(readlink -f "$(dirname "$(readlink -f "${0}")")")
cd "${REPODIR}" || exit

if [[ $(python -c 'import sys; print(sys.version_info.major)') == "3" ]]; then
  PY=python
else
  PY=python3
fi
${PY} hw2.py "$1" "$2" "$3" "$4" "$5" "$6" --generative
