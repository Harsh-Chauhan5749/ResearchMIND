#!/bin/bash
# ================================================================
# ResearchMind AI - Quick Setup Script
# ================================================================

set -e

echo "🧠 ResearchMind AI Setup"
echo "========================"

# Python check
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $python_version detected"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Copy .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "📋 Created .env from template"
    echo "⚠️  IMPORTANT: Edit .env and add your API key before starting"
    echo ""
    echo "   Option 1 (Groq - Recommended):"
    echo "     Sign up free: https://console.groq.com"
    echo "     Set: GROQ_API_KEY=your_key"
    echo ""
    echo "   Option 2 (Gemini):"
    echo "     Sign up free: https://ai.google.dev"
    echo "     Set: GEMINI_API_KEY=your_key"
    echo ""
    echo "   Option 3 (Ollama - Offline):"
    echo "     Install: https://ollama.ai"
    echo "     Run: ollama pull llama3.2"
    echo "     Set: LLM_PROVIDER=ollama"
else
    echo "✅ .env already exists"
fi

# Create data dirs
mkdir -p data/uploads data/chroma_db data/exports
echo "✅ Data directories created"

echo ""
echo "================================================================"
echo "✅ Setup complete!"
echo ""
echo "To start the app:"
echo "   streamlit run streamlit_app.py"
echo "================================================================"
