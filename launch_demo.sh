#!/bin/bash

# DP Computer Vision Pipeline - Streamlit Demo Launcher
# This script launches the interactive web interface for the pipeline

echo "🤖 DP Computer Vision Pipeline - Streamlit Demo"
echo "================================================"
echo ""

# Check if we're in the correct directory
if [ ! -f "streamlit_demo.py" ]; then
    echo "❌ Error: streamlit_demo.py not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if config file exists
if [ ! -f "configs/config.yaml" ]; then
    echo "❌ Error: configs/config.yaml not found"
    echo "Please ensure the configuration file exists"
    exit 1
fi

echo "✅ Configuration file found"
echo "✅ Streamlit demo file found"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed"
    echo "Please install it using: pip install streamlit>=1.28.0"
    exit 1
fi

echo "✅ Streamlit is installed"
echo ""

# Launch the demo
echo "🚀 Launching Streamlit demo..."
echo "📱 The web interface will open in your default browser"
echo "🔗 If it doesn't open automatically, go to: http://localhost:8501"
echo ""
echo "💡 To stop the demo, press Ctrl+C"
echo ""

# Start streamlit with custom configuration
streamlit run streamlit_demo.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.base "light" \
    --theme.primaryColor "#1f77b4" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6"
