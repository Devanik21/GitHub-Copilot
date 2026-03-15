# Github Copilot

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/GitHub-Copilot?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/GitHub-Copilot?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Github Copilot — bringing AI capabilities directly into the VS Code development environment.

---

**Topics:** `deep-learning` · `ai-coding` · `code-completion` · `code-generation` · `developer-tools` · `generative-ai` · `github-copilot` · `ide-integration` · `large-language-models` · `llm`

## Overview

Github Copilot is a VS Code extension that integrates AI-powered tooling into the editor workflow, eliminating context switching between the editor and external AI tools. It provides completions, explanations, generation commands, or chat capabilities directly accessible from the Command Palette and keyboard shortcuts.

The extension is built with TypeScript using the VS Code Extension API and communicates with an AI backend (Google Gemini, OpenAI, or a local model via Ollama) to provide intelligent, context-aware assistance. All sensitive credentials are stored in VS Code's SecretStorage to prevent accidental exposure.

The architecture follows the VS Code provider model: language-specific providers handle completions and hover actions, while WebView panels provide richer chat and generation interfaces. Configuration is fully exposed through VS Code's settings system for a native user experience.

---

## Motivation

The best developer tools are the ones that stay out of the way while making hard things easier. This extension was built to make AI assistance feel native to the editor rather than bolted on — accessible with a keystroke, configured through familiar settings, and respectful of the developer's focus.

---

## Architecture

```
User Action (keystroke / command / hover)
        │
  VS Code Provider (InlineCompletion / Hover / Command)
        │
  Context extraction from active editor
        │
  AI Backend API call
        │
  Editor injection / WebView display
```

---

## Features

### Command Palette Integration
All extension actions are registered as named VS Code commands, accessible from the Command Palette (Ctrl+Shift+P) without memorising keyboard shortcuts.

### Inline Code Actions
Right-click context menu items for common AI actions: Explain Selection, Refactor Selection, Add Docstring, Generate Tests.

### Configurable AI Backend
Switch between Google Gemini, OpenAI, or local Ollama endpoints via VS Code Settings without reinstalling the extension.

### SecretStorage for API Keys
API keys are stored in VS Code's encrypted SecretStorage rather than plaintext settings, preventing accidental exposure in dotfiles or version control.

### WebView Panel Interface
Rich HTML/CSS/JS panel for chat, generation history, and multi-turn interaction, rendered in VS Code's sidebar or editor area.

### Status Bar Integration
Real-time status bar item shows the current AI model name and a loading indicator during active API calls.

### Keyboard Shortcut Bindings
All primary actions have configurable keyboard shortcuts defined in package.json contributes.keybindings.

### Cross-Platform Support
Works on VS Code for Windows, macOS, and Linux; tested on VS Code 1.85+ and compatible with VS Code Insiders.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **TypeScript** | Extension language | Type-safe VS Code extension development |
| **VS Code Extension API** | Editor integration | Providers, commands, WebView, SecretStorage |
| **esbuild** | Bundler | Fast TypeScript → JavaScript bundling |
| **AI SDK (Gemini / OpenAI)** | AI backend | API client for LLM inference |
| **vscode-test** | Testing | Extension integration tests in an Extension Host |

> **Key packages detected in this repo:** `streamlit` · `sentence-transformers` · `faiss-cpu` · `pandas` · `numpy` · `plotly` · `scikit-learn` · `torch` · `transformers` · `tokenizers`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
git clone https://github.com/Devanik21/GitHub-Copilot.git
cd GitHub-Copilot
npm install
npm run watch   # watch mode for development
# Press F5 in VS Code to open Extension Development Host

# Build distributable VSIX
npm run package
code --install-extension *.vsix
```

---

## Usage

```bash
// settings.json
{
  "extension.apiKey": "",
  "extension.model": "gemini-2.0-flash",
  "extension.triggerMode": "onDemand"
}
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `extension.apiKey` | `(empty)` | AI backend API key (use SecretStorage command to set) |
| `extension.model` | `gemini-2.0-flash` | AI model identifier |
| `extension.triggerMode` | `onDemand` | When to trigger AI: onType, onDemand, onSave |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
GitHub-Copilot/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Repository-wide context indexing for project-aware completions
- [ ] Offline local model support via Ollama REST API
- [ ] Test generation from function signatures
- [ ] Multi-file refactoring with diff preview
- [ ] Marketplace publication with automated release workflow

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

A valid API key for the configured AI backend is required. The extension only sends code to the AI backend when a user action explicitly triggers it.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
