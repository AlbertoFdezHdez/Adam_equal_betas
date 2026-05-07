# Runtime assets

This directory is used for local runtime-only files.

Do not commit private credentials here.

Ignored local files:

- `kaggle.json`
- `hf_token.txt`

The training and dataset-preparation scripts can read these files locally, but public releases should document how users can provide their own credentials through environment variables or local files.
