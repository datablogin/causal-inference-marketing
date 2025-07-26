# Enhanced Claude Review Script

This enhanced version of `claude-review.sh` provides comprehensive PR review functionality that matches the GitHub Action capabilities.

## Features

### 🔍 **Comprehensive PR Context**
- Fetches complete PR metadata (title, author, branch, changes, commits)
- Includes full diff and file list in review context
- Provides detailed statistics about changes

### 🎯 **Focus Areas**
- `--focus security` - Security vulnerabilities and best practices
- `--focus performance` - Performance bottlenecks and optimizations  
- `--focus testing` - Test coverage and quality
- `--focus causal-inference` - Causal inference statistical validity
- `--focus style` - Code style and formatting

### 📤 **Multiple Output Modes**
- **Comment mode** (default): Posts review directly as PR comment
- **File mode**: Saves detailed markdown reviews to `reviews/manual/`
- **Draft comment mode**: Posts as draft PR comment

### 🧠 **Intelligent Prompts**
- Automatically detects causal inference files
- Generates context-aware prompts based on changed files
- Includes domain-specific review criteria

### 🔧 **Enhanced Functionality**
- **Dry run mode**: Preview what would be reviewed
- **Diff size management**: Configurable limits to prevent execution errors
- **Token usage estimation**: Rough cost tracking
- **Robust error handling**: Validates dependencies and PR existence
- **Smart file naming**: Includes focus area and timestamp

## Usage Examples

```bash
# Basic review of current PR (posts as comment)
./claude-review.sh

# Review specific PR with security focus
./claude-review.sh --focus security 54

# Save review to file instead of posting
./claude-review.sh --save-file 54

# Preview causal inference review
./claude-review.sh --dry-run --focus causal-inference 54

# Draft comment for team review
./claude-review.sh --draft-comment --focus performance 54

# Handle large PRs with diff limiting
./claude-review.sh --max-diff-lines 500 54

# No diff limit for comprehensive review
./claude-review.sh --max-diff-lines 0 54
```

## Dependencies

- **GitHub CLI** (`gh`) - For PR data and posting comments
- **Claude Code** (`claude`) - For AI-powered reviews  
- **jq** - For JSON parsing
- **Git** - For branch management

## File Organization

Reviews are saved to:
```
reviews/manual/
├── pr-54-security-20241126_1430.md
├── pr-55-causal-inference-20241126_1530.md
└── pr-56-20241126_1630.md
```

## Causal Inference Intelligence

When reviewing causal inference code, the script automatically checks for:

- ✅ Proper `BaseEstimator` inheritance patterns
- ✅ Correct usage of `TreatmentData`/`OutcomeData`/`CovariateData` models  
- ✅ Statistical assumption checking and validation
- ✅ Bootstrap implementation and confidence intervals
- ✅ Treatment of missing data and edge cases
- ✅ Reproducibility through proper random state management

## Output Format

### File Mode
Creates comprehensive markdown files with:
- PR metadata and context
- Complete diff information  
- Review prompt used
- Claude's analysis and feedback
- Token usage estimate

### Comment Mode
Posts structured comments with:
- PR context summary
- Review feedback sections
- Professional formatting for team visibility

## Migration from Original Script

The enhanced script is backward compatible:
- `./claude-review.sh` works exactly like the original
- All new features are opt-in via flags
- Preserves existing file organization

## Diff Size Management

The script includes intelligent diff handling to prevent execution errors with large PRs:

- **Default limit**: 500 lines (configurable with `--max-diff-lines`)
- **Smart truncation**: Shows first N lines with summary message
- **GitHub link**: Provides link to full diff when truncated
- **No limit option**: Use `--max-diff-lines 0` for full diff inclusion

### Large PR Handling

For PRs over 1182 lines (common execution threshold):
```bash
# Use default 500-line limit (recommended)
./claude-review.sh 54

# Increase limit for comprehensive review
./claude-review.sh --max-diff-lines 1000 54

# Focus on specific aspects to reduce scope
./claude-review.sh --focus security --max-diff-lines 300 54
```

## Integration with Development Workflow

Add to your workflow:
```bash
# Before creating PR
make ci                           # Run tests
./claude-review.sh --dry-run     # Preview review

# After creating PR  
./claude-review.sh               # Post review comment (default behavior)
```

This enhanced script brings the full power of the GitHub Action to your local development environment!