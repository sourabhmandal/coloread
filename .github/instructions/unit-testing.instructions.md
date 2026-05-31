---
description: "Use when writing or updating pytest tests, especially after git diff changes. Focus on meaningful test cases, edge cases, failure paths, and keeping touched coverage above 90%."
applyTo: "tests/**/*.py"
---
# Unit Testing Guidelines

- Start from the current git diff before adding or changing tests, and target the behavior that changed.
- Write tests that prove observable behavior, not internal implementation details.
- Include meaningful edge cases for each changed path: empty inputs, invalid types, missing config, malformed data, boundary sizes, and dependency failures.
- Cover both happy paths and failure paths when a change introduces branching, validation, parsing, or error handling.
- Prefer deterministic tests; mock external services, network calls, LLMs, and filesystem boundaries when they are not the subject of the test.
- Keep assertions specific to the contract being tested: status codes, response bodies, exceptions, side effects, and returned values.
- If a diff touches code that can be covered by tests, add or update tests until coverage for the changed area is above 90%.
- Validate the result with pytest before finishing, using the narrowest useful command for the touched code.