#!/usr/bin/env bash
set -euo pipefail

# Lint documentation for stale references.
# Run in CI or pre-commit to catch outdated docs before merge.

FAIL=0

# 1. Check for old product/repo names
echo "Checking for stale names..."
if grep -rn "mscode\|mindspore-code\|MindSpore Code\|github.com/mindspore-lab/mindspore-cli\|github.com/mindspore-lab/mindspore-skills\|mscli-server\|internal/server\|mscli.dev" --include="*.md" --include="*.go" --include="*.sh" \
    | grep -v ".git" | grep -v "superpowers" | grep -v "CHANGELOG" | grep -v "scripts/lint-docs.sh"; then
  echo "FAIL: Found stale product, repository, server, or install references"
  FAIL=1
fi

# 2. Check for old org name
if grep -rn "vigo999/mindspore" --include="*.md" --include="*.go" --include="*.sh" \
    | grep -v ".git" | grep -v "mindspore-skills" | grep -v "scripts/lint-docs.sh"; then
  echo "FAIL: Found stale 'vigo999' org references"
  FAIL=1
fi

# 3. Check markdown links point to existing files
echo "Checking markdown links..."
for md in $(find . -name "*.md" -not -path "./.git/*" -not -path "./integrations/skills/*" -not -path "./.mscli/*"); do
  # Extract relative links like [text](path/to/file.md)
  while read -r link; do
    # Skip URLs, anchors, mailto
    if [[ "$link" == http* ]] || [[ "$link" == \#* ]] || [[ "$link" == mailto* ]]; then
      continue
    fi
    # Strip anchor from link
    file="${link%%#*}"
    if [ -z "$file" ]; then
      continue
    fi
    # Resolve relative to the markdown file's directory
    dir=$(dirname "$md")
    target="$dir/$file"
    if [ ! -e "$target" ]; then
      echo "FAIL: $md links to '$link' but '$target' does not exist"
      FAIL=1
    fi
  done < <(grep -oP '\[.*?\]\(\K[^)]+' "$md" 2>/dev/null || true)
done

# 4. Check directory structure in contributor guide matches reality
echo "Checking contributor guide structure..."
if [ -f docs/agent-contributor-guide.md ]; then
  for dir in cmd/mscli internal/app internal/factory internal/issues internal/pathpolicy internal/train internal/update internal/version internal/workspacefile \
    agent/context agent/loop agent/memory agent/session integrations/domain integrations/llm integrations/skills \
    permission report runtime/artifacts runtime/mcp runtime/probes runtime/shell tools/fs tools/mcp tools/shell tools/skills ui configs; do
    if [ ! -d "$dir" ]; then
      echo "FAIL: docs/agent-contributor-guide.md references '$dir' but it doesn't exist"
      FAIL=1
    fi
  done
fi

if [ "$FAIL" -eq 1 ]; then
  echo ""
  echo "Documentation lint failed. Fix the issues above before committing."
  exit 1
fi

echo "Documentation lint passed."
