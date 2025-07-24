Please analyze and fix the GitHub issue: $ARGUMENTS.

Follow these steps:

1. Use `gh pr checks` to get the runs associated with the PR or for the current PR you are working on if no PR is specified
2. Use `gh run view` for the run ID and get the detailed list of the failures
3. Understand the test failures for each test
4. Search the codebase for the relevant files
5. Create todos for each of the failures
6. Implement the necessary changes to fix the failing tests
7. Ensure the code passes linting and type checking
8. Push the updates to PR to GitHub
9. Sleep for 120 seconds until the tests are complete and then start back at step 1
10. If the tests are not complete, sleep another 120 seconds and then start back at step 1

Remember to use the GitHub CLI (`gh`) for all GitHub-related tasks.