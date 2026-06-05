#!/usr/bin/env python3
"""Health check for the token-service container. Exit 0 = healthy, 1 = unhealthy."""
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://localhost:8080/health", timeout=5) as r:
        if r.status == 200:
            print("OK")
            sys.exit(0)
        print(f"FAIL: HTTP {r.status}")
        sys.exit(1)
except Exception as exc:
    print(f"FAIL: {exc}")
    sys.exit(1)
