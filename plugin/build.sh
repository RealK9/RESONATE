#!/bin/bash
# RESONATE Bridge — Build Script
# Builds the VST3/AU plugin using CMake + JUCE
#
# Prerequisites:
#   - CMake 3.22+
#   - Xcode command line tools
#   - Git (for JUCE FetchContent)
#
# Usage:
#   cd plugin && ./build.sh
#   Or: ./build.sh release

set -e

BUILD_TYPE="${1:-Debug}"
BUILD_DIR="build"

echo "━━━ RESONATE Bridge ━━━"
echo "Building $BUILD_TYPE..."

# Configure
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -G "Unix Makefiles"

# Build
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$(sysctl -n hw.ncpu)"

echo ""
echo "✓ Build complete!"

# Install plugins to system directories
mkdir -p ~/Library/Audio/Plug-Ins/VST3 ~/Library/Audio/Plug-Ins/Components
cp -R "$BUILD_DIR/ResonateBridge_artefacts/$BUILD_TYPE/VST3/RESONATE Bridge.vst3" ~/Library/Audio/Plug-Ins/VST3/
cp -R "$BUILD_DIR/ResonateBridge_artefacts/$BUILD_TYPE/AU/RESONATE Bridge.component" ~/Library/Audio/Plug-Ins/Components/

echo "  VST3: ~/Library/Audio/Plug-Ins/VST3/RESONATE Bridge.vst3"
echo "  AU:   ~/Library/Audio/Plug-Ins/Components/RESONATE Bridge.component"
echo ""
echo "Add RESONATE Bridge to your DAW master channel to sync."
