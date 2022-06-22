# HandbookForDatascience

Handbook For Datascience

# Setup

```bash
# pipenv users
pipenv lock --dev && pipenv install --dev

# other users
pipenv install requirements.txt 
# or for window users
pipenv install requirements_win.txt 
```

## vscode user settings

* install the extenstion (vscode-yaml)[https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml]
* (optional) add the following settings in `settings.json`

    ```json
    {
        "yaml.schemas": {
            "https://squidfunk.github.io/mkdocs-material/schema.json": "mkdocs.yml"
        }
    }
    ```

## helpful documents

- [https://www.mkdocs.org/](https://www.mkdocs.org/)
- [https://squidfunk.github.io/mkdocs-material/](https://squidfunk.github.io/mkdocs-material/)