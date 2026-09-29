# DNB Chinese Text Scoring

Chinese category-conditioned score regression using `hfl/chinese-macbert-base`, with a local scoring API and a GitHub Pages frontend.

- Website: https://chizhang-805.github.io/rhythm-dnb-text-zc/
- Chinese instructions: [README_CN.md](README_CN.md)
- Public repository: code and tests only. Research datasets, user text, model weights, experiment outputs and recovery archives are excluded.

## Website Status

GitHub Pages serves the static UI, not the Python model. The public page shows all categories and their fixed outputs. Scoring remains disabled until a trained model is provided through an explicitly configured HTTPS API. It never generates placeholder or random scores.

The local app continues to use its same-origin Python service:

```sh
python run.py serve --port 8765
```

## Training and Tests

Install `requirements.txt` in a Python environment with an appropriate PyTorch build. Restore the private dataset separately to `dataset/master.sqlite` before preparing or training the model.

```sh
python run.py test
python run.py prepare
python run.py train
python run.py evaluate --split test
```

Unit tests use disposable fixtures; the private-corpus integration check is skipped in a code-only clone. No model downloads are needed for unit tests.

## Pages Deployment

Pushes to `main` run `.github/workflows/pages.yml`. Only four allowlisted frontend assets, the public output schema, a deployment config and `.nojekyll` are published. Build checks reject unexpected files in the publication directory.

```sh
python scripts/build_pages.py
```

By default, no inference endpoint is configured and no input text is sent anywhere. To connect a separately secured inference deployment later, set the repository Actions variable `DNB_PUBLIC_API_URL` to its HTTPS base URL and rerun deployment. That service must implement `GET /api/health` and `POST /api/predict`, allow the exact Pages origin through CORS, and provide appropriate authentication, rate limits and privacy controls. The included loopback-only development server is not an internet-facing production deployment.

Never put an API key, private dataset, model weights or example patient text in a Pages artifact or frontend config.
