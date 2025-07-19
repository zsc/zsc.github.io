# arXiv Paper Summary Generator - Technical Overview

## Overview
A web-based tool that fetches arXiv papers and generates summaries using Google's Gemini API. The application features a tabbed interface with persistent storage.

## Key Features

### 1. Tab Management
- **Multi-tab interface**: Each arXiv paper opens in its own tab
- **Tab persistence**: Uses localStorage to cache all tabs across page reloads
- **Unread indicators**: Visual indicators (red dots) for new summaries
- **Favicon notifications**: Shows unread count in browser tab

### 2. PDF Processing
- Fetches PDFs from arXiv using CORS proxy (corsproxy.io)
- Converts PDF to base64 for Gemini API processing
- Supports both Gemini 2.5 Pro and Flash models

### 3. Markdown Rendering
- Uses marked.js for Markdown parsing
- KaTeX integration for LaTeX math rendering (both inline `$...$` and block `$$...$$`)
- Special handling to escape `_` and `*` within math expressions

### 4. Image Reference Processing
- Optional feature to increment image filenames (x1.png → x2.png)
- Useful for academic papers with figure references
- Toggle per tab, state persisted in localStorage

### 5. Error Handling & Recovery
- Retry mechanism with 3 attempts for network failures
- "Retry" button for failed summaries
- Graceful handling of interrupted API calls (page closure during processing)
- Converts stale "loading" states to errors on page reload

### 6. Storage Implementation
- Uses localStorage for tab persistence (~5MB capacity)
- Stores: paper content, status, settings, unread state
- Handles quota exceeded errors gracefully
- External image references keep storage usage minimal

### 7. User Experience
- Real-time status updates during processing
- Download summaries as Markdown files
- Customizable prompts via prompts.js
- API key can be passed via URL parameter

## Technical Details

### State Management
```javascript
papers = {
  "arxiv_id": {
    status: "loading|success|error",
    summary: "markdown content",
    error: "error message",
    isUnread: boolean,
    incrementImages: boolean
  }
}
```

### Storage Keys
- `arxiv_tabs_cache`: Main tab data storage
- `selectedPromptIndex`: User's selected prompt template

### API Integration
- Uses Google Generative AI SDK via ES modules
- Supports concurrent paper processing
- Base64 encoding for PDF data transmission

## Security Considerations
- API keys stored in memory only (not persisted)
- External link handling with `target="_blank"` and `rel="noopener noreferrer"`
- No server-side component - fully client-side application