#!/bin/bash

# Colors for Discord embeds
SUCCESS_COLOR=3066993
FAILURE_COLOR=15158332

# Get the webhook URL from environment variable
WEBHOOK_URL=$DISCORD_WEBHOOK

# Set status based on test results
if [ "$1" == "success" ]; then
    STATUS_MESSAGE="✅ Tests Passed"
    EMBED_COLOR=$SUCCESS_COLOR
else
    STATUS_MESSAGE="❌ Tests Failed"
    EMBED_COLOR=$FAILURE_COLOR
fi

# Get git information
AUTHOR_NAME="$(git log -1 --pretty="%aN")"
COMMITTER_NAME="$(git log -1 --pretty="%cN")"
COMMIT_SUBJECT="$(git log -1 --pretty="%s")"
COMMIT_MESSAGE="$(git log -1 --pretty="%b")"
COMMIT_SHA="$(git rev-parse --short HEAD)"
REPOSITORY="$(git config --get remote.origin.url | sed 's/.*\/\([^ ]*\/[^.]*\).*/\1/')"

# Format credits
if [ "$AUTHOR_NAME" == "$COMMITTER_NAME" ]; then
    CREDITS="$AUTHOR_NAME authored & committed"
else
    CREDITS="$AUTHOR_NAME authored & $COMMITTER_NAME committed"
fi

# Get current timestamp
TIMESTAMP=$(date --utc +%FT%TZ)

# Prepare webhook data
WEBHOOK_DATA='{
    "embeds": [{
        "color": '$EMBED_COLOR',
        "author": {
            "name": "'"$STATUS_MESSAGE"' - '"$REPOSITORY"'"
        },
        "title": "'"$COMMIT_SUBJECT"'",
        "description": "'"${COMMIT_MESSAGE//$'\n'/ }"\\n\\n"$CREDITS"'",
        "fields": [
            {
                "name": "Commit",
                "value": "'"$COMMIT_SHA"'",
                "inline": true
            },
            {
                "name": "Branch",
                "value": "'"$(git rev-parse --abbrev-ref HEAD)"'",
                "inline": true
            }
        ],
        "timestamp": "'"$TIMESTAMP"'"
    }]
}'

# Send webhook
echo -e "[Webhook]: Sending webhook to Discord...\n"

curl --fail --progress-bar \
    -A "GitHub-Actions-Webhook" \
    -H "Content-Type: application/json" \
    -d "$WEBHOOK_DATA" \
    "$WEBHOOK_URL" \
    && echo -e "\n[Webhook]: Successfully sent the webhook." \
    || echo -e "\n[Webhook]: Unable to send webhook."