Please analyze and fix the GitHub issue: $ARGUMENTS.

Follow these steps:

1. Use `gh pr view` to get the PR comments by Claude for the PR in arguments, or for the current PR you are working on if no PR is specified
2. Understand the list of recommendations described in the PR by the commenter Claude
3. Search the codebase for the relevant files
4. Create todos for each of the recommendations
5. Implement the necessary changes to fix the issue
6. Write and run tests to verify the fix
7. Ensure the code passes linting and type checking
8. Create a descriptive commit message
9. Push the updates to PR to GitHub

Remember to use the GitHub CLI (`gh`) for all GitHub-related tasks.