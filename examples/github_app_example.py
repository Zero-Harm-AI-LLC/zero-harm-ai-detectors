#!/usr/bin/env python3
"""
Zero Harm AI Detectors - GitHub App Integration Example

Shows how to integrate the library into a GitHub App that reviews
PRs for sensitive data leakage (API keys, PII, etc.).

File: examples/github_app_example.py
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import json

from zero_harm_ai_detectors import detect, AI_AVAILABLE


@dataclass
class PRFile:
    """A file changed in a PR."""
    filename: str
    patch: str
    additions: int = 0


@dataclass
class ScanResult:
    """Result of scanning a single file."""
    filename: str
    has_issues: bool
    detections: List[Dict[str, Any]]
    severity: str


class PRScanner:
    """Scans PR diffs for sensitive data."""
    
    # Types that should block a PR
    BLOCKING_TYPES = {"API_KEY", "SECRET", "SSN", "CREDIT_CARD"}
    
    def __init__(self, use_ai: bool = False):
        """
        Initialize scanner.
        
        Args:
            use_ai: If True and AI available, uses mode='ai'
        """
        self.mode = "ai" if (use_ai and AI_AVAILABLE) else "regex"
    
    def scan_file(self, file: PRFile) -> ScanResult:
        """Scan a single file's diff."""
        # Only scan additions (new code)
        additions = self._extract_additions(file.patch)
        
        if not additions.strip():
            return ScanResult(
                filename=file.filename,
                has_issues=False,
                detections=[],
                severity="none",
            )
        
        # Run detection
        result = detect(
            additions,
            mode=self.mode,
            detect_pii=True,
            detect_secrets=True,
            detect_harmful=False,  # Not needed for code
        )
        
        # Convert detections
        detections = [
            {
                "type": d.type,
                "text": d.text[:50] + "..." if len(d.text) > 50 else d.text,
                "confidence": d.confidence,
            }
            for d in result.detections
        ]
        
        # Determine severity
        types = {d.type for d in result.detections}
        if types & {"API_KEY", "SECRET"}:
            severity = "critical"
        elif types & {"SSN", "CREDIT_CARD"}:
            severity = "high"
        elif types & {"EMAIL", "PHONE"}:
            severity = "medium"
        elif types:
            severity = "low"
        else:
            severity = "none"
        
        return ScanResult(
            filename=file.filename,
            has_issues=len(detections) > 0,
            detections=detections,
            severity=severity,
        )
    
    def scan_pr(self, pr_number: int, files: List[PRFile]) -> Dict[str, Any]:
        """Scan all files in a PR."""
        results = [self.scan_file(f) for f in files]
        
        files_with_issues = sum(1 for r in results if r.has_issues)
        total_detections = sum(len(r.detections) for r in results)
        
        # Check if PR should be blocked
        should_block = any(
            any(d["type"] in self.BLOCKING_TYPES for d in r.detections)
            for r in results
        )
        
        # Overall severity
        severities = [r.severity for r in results]
        if "critical" in severities:
            overall = "critical"
        elif "high" in severities:
            overall = "high"
        elif "medium" in severities:
            overall = "medium"
        elif "low" in severities:
            overall = "low"
        else:
            overall = "none"
        
        return {
            "pr_number": pr_number,
            "mode_used": self.mode,
            "files_scanned": len(files),
            "files_with_issues": files_with_issues,
            "total_detections": total_detections,
            "severity": overall,
            "should_block": should_block,
            "file_results": [
                {
                    "filename": r.filename,
                    "has_issues": r.has_issues,
                    "severity": r.severity,
                    "detections": r.detections,
                }
                for r in results
            ],
        }
    
    def _extract_additions(self, patch: str) -> str:
        """Extract added lines from diff patch."""
        lines = []
        for line in patch.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                lines.append(line[1:])
        return "\n".join(lines)


def main():
    print("=" * 70)
    print("Zero Harm AI Detectors - GitHub App Integration Example")
    print("=" * 70)
    
    # Sample PR files
    pr_files = [
        PRFile(
            filename="config.py",
            patch="""
@@ -1,3 +1,8 @@
 # Config
+API_KEY = "sk-1234567890abcdef1234567890abcdef"
+AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
 DB_URL = os.environ.get("DATABASE_URL")
+# Contact: john@company.com
            """,
        ),
        PRFile(
            filename="utils.py",
            patch="""
@@ -10,3 +10,6 @@
 def helper():
     pass
+
+# Debug: SSN 123-45-6789
+# Phone: 555-123-4567
            """,
        ),
        PRFile(
            filename="README.md",
            patch="""
@@ -1,2 +1,4 @@
 # Project
+
+Contact: john@example.com
            """,
        ),
    ]
    
    # ================================================================
    # Example 1: Regex Mode Scan
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 1: Regex Mode PR Scan")
    print("-" * 70)
    
    scanner = PRScanner(use_ai=False)
    result = scanner.scan_pr(pr_number=123, files=pr_files)
    
    print(f"PR #{result['pr_number']} Results:")
    print(f"  Mode: {result['mode_used']}")
    print(f"  Files: {result['files_scanned']} scanned, {result['files_with_issues']} with issues")
    print(f"  Detections: {result['total_detections']}")
    print(f"  Severity: {result['severity']}")
    print(f"  Should Block: {result['should_block']}")
    
    print("\nFile Details:")
    for fr in result['file_results']:
        if fr['has_issues']:
            print(f"  ⚠️  {fr['filename']} ({fr['severity']})")
            for d in fr['detections']:
                print(f"      - {d['type']}: {d['text']}")
        else:
            print(f"  ✓  {fr['filename']}")
    
    # ================================================================
    # Example 2: AI Mode Scan (if available)
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 2: AI Mode PR Scan")
    print("-" * 70)
    
    scanner_ai = PRScanner(use_ai=True)
    result_ai = scanner_ai.scan_pr(pr_number=123, files=pr_files)
    
    print(f"Mode used: {result_ai['mode_used']}")
    print(f"Total detections: {result_ai['total_detections']}")
    
    if result_ai['mode_used'] == 'ai':
        print("(AI mode may detect additional person names)")
    else:
        print("(AI not available, fell back to regex)")
    
    # ================================================================
    # Example 3: Generate Review Comment
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 3: Generate GitHub Review Comment")
    print("-" * 70)
    
    def generate_comment(scan_result: Dict[str, Any]) -> str:
        """Generate a GitHub PR review comment."""
        if not any(fr['has_issues'] for fr in scan_result['file_results']):
            return "✅ No sensitive data detected in this PR."
        
        lines = []
        if scan_result['should_block']:
            lines.append("🚫 **This PR contains sensitive data that must be removed:**")
        else:
            lines.append("⚠️ **This PR may contain sensitive data:**")
        
        for fr in scan_result['file_results']:
            if fr['has_issues']:
                lines.append(f"\n**{fr['filename']}** ({fr['severity']}):")
                for d in fr['detections']:
                    lines.append(f"  - {d['type']}: `{d['text']}`")
        
        if scan_result['should_block']:
            lines.append("\n❌ Please remove the sensitive data before merging.")
        
        return "\n".join(lines)
    
    comment = generate_comment(result)
    print("Generated comment:")
    print("-" * 40)
    print(comment)
    print("-" * 40)
    
    # ================================================================
    # Example 4: Webhook Handler Pattern
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 4: Webhook Handler Pattern")
    print("-" * 70)
    
    print("""
    @app.post("/webhook")
    def github_webhook(payload: dict):
        pr_number = payload["pull_request"]["number"]
        is_paid = check_subscription(payload["installation"]["id"])
        
        scanner = PRScanner(use_ai=is_paid)
        files = fetch_pr_files(payload)  # From GitHub API
        
        result = scanner.scan_pr(pr_number, files)
        
        # Post review via GitHub API
        github.post_review(
            repo=payload["repository"]["full_name"],
            pr_number=pr_number,
            event="REQUEST_CHANGES" if result["should_block"] else "COMMENT",
            body=generate_comment(result),
        )
    """)
    
    # ================================================================
    # Example 5: JSON Output for API
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 5: JSON Output for API")
    print("-" * 70)
    
    print("API response format:")
    print(json.dumps(result, indent=2)[:500] + "...")
    
    print("\n" + "=" * 70)
    print("GitHub App integration examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
