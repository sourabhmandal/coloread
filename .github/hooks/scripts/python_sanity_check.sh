#!/bin/sh

cat <<'JSON'
{
  "hookSpecificOutput": {
    "hookEventName": "Stop",
    "systemMessage": "Python/FastAPI sanity check: before ending, review request and response models, type hints, error messages, dependency wiring, package versions, file layout, and tests. Call out fragile control flow, weak validation, missing dependency boundaries, or non-idiomatic code in this repo's router, schemas, settings, and assistant modules."
  }
}
JSON