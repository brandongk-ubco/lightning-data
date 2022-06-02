#!/usr/bin/env bash

mypy lightningdata
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count  --max-complexity=15 --statistics
yapf --diff -r lightningdata