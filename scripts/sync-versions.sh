#!/usr/bin/env bash
#
# Point the deployment manifests at a released version of sys-stats.
#
# The package version itself is derived from git tags (hatch-vcs), so nothing
# here decides what the version is: this only propagates an already-released
# one to the three files that must name an image tag a human can pull, and that
# therefore cannot derive anything at build time.
#
#   compose.yaml     image: obeoneorg/sys-stats:X.Y.Z
#   README.md        docker run ... ghcr.io/obeone/sys-stats:X.Y.Z
#   chart/Chart.yaml appVersion: "X.Y.Z", plus a patch bump of the chart's own
#                    version, since changing what the chart deploys is a change
#                    to the chart.
#
# The release workflow runs this on main after publishing X.Y.Z; running it by
# hand is fine too, and a no-op when everything already points at the argument.
#
# Usage: scripts/sync-versions.sh X.Y.Z
set -euo pipefail

version="${1:-}"
if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "usage: $0 X.Y.Z" >&2
    exit 64
fi

# Run from the repository root whatever the caller's working directory.
cd "$(dirname "$0")/.."

# perl rather than sed -i: BSD sed (macOS) and GNU sed disagree on whether -i
# takes an argument, and this script has to run on both.
perl -pi -e "s{(image: obeoneorg/sys-stats:)\S+}{\${1}$version}" compose.yaml
perl -pi -e "s{(ghcr\.io/obeone/sys-stats:)[0-9][^\s\`]*}{\${1}$version}" README.md

current_app="$(perl -ne 'print $1 if /^appVersion:\s*"?([^"\s]+)"?/' chart/Chart.yaml)"
if [[ $current_app != "$version" ]]; then
    # The chart's own version moves independently of the application's, but it
    # does have to move: two charts both calling themselves 0.2.0 while
    # deploying different images is exactly what chart versions exist to stop.
    chart_version="$(perl -ne 'print $1 if /^version:\s*(\S+)/' chart/Chart.yaml)"
    IFS=. read -r major minor patch <<<"$chart_version"
    next_chart="$major.$minor.$((patch + 1))"

    perl -pi -e "s{^version:.*}{version: $next_chart}" chart/Chart.yaml
    perl -pi -e "s{^appVersion:.*}{appVersion: \"$version\"}" chart/Chart.yaml
    echo "chart $chart_version -> $next_chart (appVersion $current_app -> $version)"
fi

echo "manifests point at $version"
