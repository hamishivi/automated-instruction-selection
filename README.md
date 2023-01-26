# python-project-template

This is a template repository for Python-based research projects.

## Usage

1. [Create a new repository](https://github.com/allenai/python-project-template/generate) from this template with the desired name of your Python project.

2. Change the name of the `my_project` directory to the name of your repo / Python project.

3. Replace all mentions of `my_project` throughout this repository with the new name.

    On OS X, a quick way to find all mentions of `my_project` is:

    ```bash
    find . -type f -not -path './.git/*' -not -path ./README.md -not -path './docs/build/*' -not -path '*__pycache__*' | xargs grep 'my_project'
    ```

    There is also a one-liner to find and replace all mentions `my_project` with `actual_name_of_project`:

    ```bash
    find . -type f -not -path './.git/*' -not -path ./README.md -not -path './docs/build/*' -not -path '*__pycache__*' -exec sed -i '' -e 's/my_project/actual_name_of_project/' {} \;
    ```

4. Update the README.md.

5. Commit and push your changes, then make sure all CI checks pass.
