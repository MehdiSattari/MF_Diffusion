#!/bin/bash
# Run ON Alvis (has the GitHub SSH key). Stages all tracked+new code, shows the
# summary, commits with the given message, and pushes to origin/main.
#   bash git_commit.sh "your message"
set -euo pipefail
cd "$(dirname "$0")"
MSG="${1:-WIP: sync from Alvis}"
git add -A
echo "---- staged changes ----"
git status --short
echo "------------------------"
git commit -m "$MSG"
git push origin main
echo "pushed to origin/main"
