#!/bin/bash

# Get arguments
TEST_OUTPUT="$1"
COVERAGE_REPORT="$2"
REPOSITORY="$3"
RUN_ID="$4"

# Color for test results (orange for warning)
EMBED_COLOR=16737095

# Get git information
AUTHOR_NAME="$(git log -1 --pretty="%aN")"
COMMITTER_NAME="$(git log -1 --pretty="%cN")"
COMMIT_SUBJECT="$(git log -1 --pretty="%s")"
COMMIT_SHA="$(git rev-parse --short HEAD)"
BRANCH="${GITHUB_REF#refs/heads/}"

# Generate GitHub URLs
ACTIONS_URL="https://github.com/$REPOSITORY/actions/runs/$RUN_ID"

# Format credits
if [ "$AUTHOR_NAME" == "$COMMITTER_NAME" ]; then
    CREDITS="$AUTHOR_NAME authored & committed"
else
    CREDITS="$AUTHOR_NAME authored & $COMMITTER_NAME committed"
fi

# Parse coverage from the coverage report
COVERAGE_PERCENT=$(echo "$COVERAGE_REPORT" | grep -oP "TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+\K\d+(?=%)") || COVERAGE_PERCENT="N/A"

# Extract failed tests
FAILED_TESTS=$(echo "$TEST_OUTPUT" | grep -B 1 "FAILED" | grep "test_" || echo "None")

# Format failed tests for Discord
if [ "$FAILED_TESTS" == "None" ]; then
    TEST_STATUS="✅ All Tests Passed"
    TEST_DETAILS="All tests executed successfully."
else
    TEST_STATUS="❌ Tests Failed"
    TEST_DETAILS="Failed Tests:\n\`\`\`\n$FAILED_TESTS\n\`\`\`"
fi

# Get current timestamp
TIMESTAMP=$(date --utc +%FT%TZ)

# Prepare webhook data
WEBHOOK_DATA='{
    "embeds": [{
        "color": '$EMBED_COLOR',
        "author": {
            "name": "Test Results - '"$REPOSITORY"'",
            "url": "'"$ACTIONS_URL"'"
        },
        "title": "'"$TEST_STATUS"'",
        "url": "'"$ACTIONS_URL"'",
        "description": "'"${TEST_DETAILS}"'\n\nCode Coverage: **'"${COVERAGE_PERCENT}"'%**\n\n'"$CREDITS"'",
        "fields": [
            {
                "name": "Commit",
                "value": "'"$COMMIT_SHA"'",
                "inline": true
            },
            {
                "name": "Branch",
                "value": "'"$BRANCH"'",
                "inline": true
            },
            {
                "name": "Details",
                "value": "[View Run Details]('"$ACTIONS_URL"')",
                "inline": true
            }
        ],
        "timestamp": "'"$TIMESTAMP"'"
    }]
}'

# Send webhook
curl --fail --progress-bar \
    -A "GitHub-Actions-Webhook" \
    -H "Content-Type: application/json" \
    -d "$WEBHOOK_DATA" \
    "$DISCORD_WEBHOOK" \
    && echo -e "\n[Webhook]: Successfully sent the webhook." \
    || echo -e "\n[Webhook]: Unable to send webhook."