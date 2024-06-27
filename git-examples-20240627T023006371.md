Sure, here are the steps to amend a commit if a reviewer requests a change:

### Step-by-Step Guide to Amend a Commit

1. **Make the Requested Changes:**
   - Edit the files as needed using your preferred text editor. For example, if you use `vi`:
     ```bash
     vi filename
     ```

2. **Add the Changes:**
   ```bash
   git add filename
   ```

3. **Amend the Last Commit:**
   - Use `git commit --amend` to add the changes to the previous commit. This will open your default text editor (e.g., `vi`) for the commit message:
     ```bash
     git commit --amend
     ```
   - Update the commit message if necessary. Ensure the `Signed-off-by:` line is included for DCO signoff. For example:
     ```
     Fix issue with feature X

     This commit fixes the issue with feature X by addressing Y.

     Signed-off-by: Your Name <your-email@example.com>
     ```
   - Save and close `vi`:
     - Press `Esc` to exit insert mode.
     - Type `:wq` and press `Enter` to save and quit.

4. **Force Push the Amended Commit:**
   - Since you have amended a commit that has already been pushed, you need to force push the changes to your branch:
     ```bash
     git push --force origin your-branch-name
     ```

### Additional Tips

- **Rebase for Multiple Commits:**
  - If you need to amend a commit that is not the last one, you can use an interactive rebase:
    ```bash
    git rebase -i HEAD~n
    ```
    Replace `n` with the number of commits you want to rebase. This opens an interactive editor where you can mark the commit you want to amend with `edit`.
  - Make your changes, then:
    ```bash
    git add filename
    git commit --amend
    git rebase --continue
    ```
  - Finally, force push your changes:
    ```bash
    git push --force origin your-branch-name
    ```

- **Addressing Merge Conflicts:**
  - If you encounter merge conflicts during a rebase, Git will pause and allow you to resolve them. After resolving conflicts, add the resolved files:
    ```bash
    git add resolved-file
    git rebase --continue
    ```

By following these steps, you can effectively amend commits in response to reviewer feedback and ensure your changes are properly integrated into the pull request.