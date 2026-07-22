#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VERSION="${1:-}"

if [ -z "${VERSION}" ]; then
  echo "Usage: ./scripts/build-release-assets.sh <version>"
  echo "Example: ./scripts/build-release-assets.sh v0.5.0-beta.2"
  exit 1
fi

if [[ "${VERSION}" != v* ]]; then
  echo "Error: version must include a leading v, for example v0.5.0-beta.2" >&2
  exit 1
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

need_cmd go
need_cmd sha256sum
PLATFORMS=(
  "linux/amd64"
  "linux/arm64"
  "darwin/amd64"
  "darwin/arm64"
  "windows/amd64"
)

GITCODE_OWNER="${GITCODE_OWNER:-mindspore}"
GITCODE_REPO="${GITCODE_REPO:-mscli}"
DIST_DIR="${MSCLI_DIST_DIR:-${REPO_ROOT}/dist}"
MODULE_PATH="$(cd "${REPO_ROOT}" && go list -m)"

echo "Building ${VERSION} into ${DIST_DIR}"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

cd "${REPO_ROOT}"

ASSETS=()
for platform in "${PLATFORMS[@]}"; do
  GOOS="${platform%/*}"
  GOARCH="${platform#*/}"
  output="mscli-${GOOS}-${GOARCH}"
  if [ "${GOOS}" = "windows" ]; then
    output="${output}.exe"
  fi
  ASSETS+=("${output}")
  echo "  -> ${output}"
  GOOS="${GOOS}" GOARCH="${GOARCH}" go build \
    -ldflags "-X ${MODULE_PATH}/internal/version.Version=${VERSION}" \
    -o "${DIST_DIR}/${output}" \
    ./cmd/mscli/
done

{
  echo "{"
  echo "  \"latest\": \"${VERSION}\","
  echo "  \"min_allowed\": \"\","
  echo "  \"download_base\": \"https://gitcode.com/${GITCODE_OWNER}/${GITCODE_REPO}/releases/download\","
  echo "  \"checksums\": {"
  last_index=$((${#ASSETS[@]} - 1))
  for index in "${!ASSETS[@]}"; do
    asset="${ASSETS[index]}"
    checksum="$(sha256sum "${DIST_DIR}/${asset}" | cut -d ' ' -f 1)"
    comma=","
    if [ "${index}" -eq "${last_index}" ]; then
      comma=""
    fi
    printf '    "%s": "%s"%s\n' "${asset}" "${checksum}" "${comma}"
  done
  echo "  }"
  echo "}"
} > "${DIST_DIR}/manifest.json"

cp "${SCRIPT_DIR}/install.sh" "${DIST_DIR}/install.sh"
chmod +x "${DIST_DIR}/install.sh"

echo ""
echo "Release assets ready in ${DIST_DIR}:"
ls -lh "${DIST_DIR}"
