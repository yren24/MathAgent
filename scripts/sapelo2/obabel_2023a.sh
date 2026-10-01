#!/bin/bash
set -euo pipefail

module purge
module load OpenBabel/3.1.1-gompi-2023a
exec obabel "$@"
