Please analyze and fix the GitHub issue: $ARGUMENTS.

Follow these steps:

1. Use `gh issue view` to get the issue details
2. Understand the problem described in the issue
3. Search the codebase for relevant files
4. Check out main branch of the repo and create a branch for the new work, include the issue number and the title in the title of the branch
5. Implement the necessary changes to solve the problem or create the feature outlined in the issue
6. Write and run tests to verify the fix
7. Ensure the code passes linting, type checking, ruff and CI/CD tests
8. Create a descriptive commit message
9. Push and create a PR, push the PR to github
10. After the PR is successfully pushed, run ./claude-review.sh on the PR number that was pushed

Remember to use the GitHub CLI (`gh`) for all GitHub-related tasks.