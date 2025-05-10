#!/bin/bash

FILE=$1
MESSAGE=$2

if [ -z "$FILE" ] || [ -z "$MESSAGE" ]; then
  echo "Usage: ./gitpush_file.sh <file_path> \"commit message\""
  exit 1
fi

if [ ! -f "$FILE" ]; then
  echo "Error: File '$FILE' does not exist."
  exit 1
fi

git add "$FILE"
git commit -m "$MESSAGE"
git push origin main
