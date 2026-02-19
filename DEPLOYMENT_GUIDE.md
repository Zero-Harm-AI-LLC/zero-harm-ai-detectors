# Deployment Guide

Complete guide for deploying Zero Harm AI Detectors.

## Pre-Deployment Checklist

- [ ] Python 3.8+ installed
- [ ] 4 GB+ RAM available (AI mode)
- [ ] 2 GB+ disk space (AI mode model cache)
- [ ] Git repository backed up

## Step 1: Publish the Library

### Run tests locally

```bash
# Base tests (no ML dependencies)
pytest tests/test_core_and_regex.py -v

# AI tests (requires pip install zero_harm_ai_detectors[ai])
pytest tests/test_ai_detectors.py -v
```

### Create a release tag

```bash
git add .
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

This triggers the `publish-to-pypi.yml` workflow which:

1. Runs `test-basic` (regex + core pattern tests)
2. Runs `test-ai` (transformer tests, with model caching)
3. Builds and publishes to PyPI
4. Creates a GitHub Release
5. Waits 2 minutes for PyPI propagation
6. Dispatches a `library-updated` event to the backend repo

### Monitor the workflow

```bash
gh run watch
# or check https://github.com/Zero-Harm-AI-LLC/zero-harm-ai-detectors/actions
```

## Step 2: Backend Auto-Update

When the library is published the backend workflow automatically:

1. Updates `requirements.txt` with the new version
2. Runs tests against the new library
3. Commits the change
4. Triggers a Render deployment

### Manual backend update (if automation fails)

```bash
cd zero-harm-ai-backend

# Update version pin
sed -i 's/zero_harm_ai_detectors>=.*/zero_harm_ai_detectors>=X.Y.Z/' requirements.txt

pip install -r requirements.txt
pytest tests/

git add requirements.txt
git commit -m "Bump zero_harm_ai_detectors to vX.Y.Z"
git push origin main
```

## Step 3: Verify Render Deployment

### Health check

```bash
curl https://zero-harm-ai-backend.onrender.com/api/health_check
# Expected: "Zero Harm AI Flask backend is running."
```

### Smoke test — PII detection

```bash
curl -X POST https://zero-harm-ai-backend.onrender.com/api/check_privacy \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact John Smith at john@example.com"}'
```

Expected response:

```json
{
  "redacted": "Contact [REDACTED_PERSON] at [REDACTED_EMAIL]",
  "detections": {
    "PERSON": [{"span": "John Smith", "start": 8, "end": 18}],
    "EMAIL": [{"span": "john@example.com", "start": 22, "end": 38}]
  }
}
```

### Performance check

```bash
time curl -X POST https://zero-harm-ai-backend.onrender.com/api/check_privacy \
  -H "Content-Type: application/json" \
  -d '{"text": "Test text"}'
# Target: < 500ms
```

## Validation

### Integration test script

```python
# test_integration.py
import requests

BASE = "https://zero-harm-ai-backend.onrender.com"

def test_email():
    r = requests.post(f"{BASE}/api/check_privacy", json={"text": "alice@example.com"})
    assert r.status_code == 200
    assert "EMAIL" in r.json()["detections"]
    print("✅ Email detection")

def test_secrets():
    r = requests.post(f"{BASE}/api/check_privacy",
                      json={"text": "key = sk-1234567890abcdef1234567890abcdef"})
    assert r.status_code == 200
    data = r.json()["detections"]
    assert "API_KEY" in data or "SECRET" in data
    print("✅ Secrets detection")

def test_person_ai():
    r = requests.post(f"{BASE}/api/check_privacy",
                      json={"text": "Contact Sarah Johnson", "mode": "ai"})
    assert r.status_code == 200
    assert "PERSON" in r.json()["detections"]
    print("✅ AI person detection")

if __name__ == "__main__":
    test_email()
    test_secrets()
    test_person_ai()
    print("\n🎉 All integration tests passed!")
```

```bash
python test_integration.py
```

### Performance test script

```python
# test_performance.py
import requests, time

BASE = "https://zero-harm-ai-backend.onrender.com"
texts = ["Contact john@example.com", "Phone: 555-123-4567", "Meet Jane at Google"] * 10

start = time.time()
for text in texts:
    assert requests.post(f"{BASE}/api/check_privacy", json={"text": text}).status_code == 200
elapsed = time.time() - start

avg_ms = elapsed / len(texts) * 1000
print(f"Processed {len(texts)} requests in {elapsed:.2f}s — avg {avg_ms:.0f}ms")
assert avg_ms < 500, "Performance target missed (< 500ms)"
print("✅ Performance test passed")
```

## Performance Tuning

### Use regex mode for high-volume scanning

```python
result = detect(text, mode="regex")   # 1–5ms vs 50–200ms for AI
```

### Skip unused detectors

```python
# Code scanning — no harmful content check needed
result = detect(diff_text, detect_harmful=False)
```

### Increase Gunicorn workers

In `render.yaml` or `Procfile`:

```yaml
startCommand: gunicorn app:app --workers 4 --timeout 120
```

### Use GPU for AI mode (if available)

```python
from zero_harm_ai_detectors import AIConfig
config = AIConfig(device="cuda")
result = detect(text, mode="ai", ai_config=config)
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ImportError: No module named 'transformers'` | `pip install 'zero_harm_ai_detectors[ai]'` |
| `RuntimeError: CUDA out of memory` | Use `AIConfig(device="cpu")` |
| Response time > 1s | Set `detect_harmful=False` or use `mode="regex"` |
| `OSError: Can't load model` | `rm -rf ~/.cache/huggingface && pip install --upgrade transformers torch` |

## Success Metrics

| Metric | Target |
|--------|--------|
| Person name accuracy (AI mode) | > 85% |
| Average response time | < 500ms |
| Uptime | > 99.5% |
| Detection coverage | > 90% |

## Post-Deployment Checklist

- [ ] Health check passing
- [ ] Integration tests passing
- [ ] Performance within target
- [ ] No errors in Render logs (`render logs zero-harm-ai-backend --tail`)
- [ ] CHANGELOG.md updated
