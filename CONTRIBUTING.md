# Contributing to DuAncntt_Project

Thank you for your interest in contributing to this recommendation system project! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear description of the bug
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, PyTorch version, etc.)
- Any relevant error messages or logs

### Suggesting Enhancements

We welcome suggestions for new features or improvements! Please open an issue with:
- A clear description of the enhancement
- Use cases and benefits
- Possible implementation approach (if applicable)

### Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/Nguyen3007/DuAncntt_Project.git
   cd DuAncntt_Project
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clean, readable code
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test Your Changes**
   - Ensure your code works as expected
   - Test with different datasets if possible
   - Verify that existing functionality still works

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```
   
   Commit message format:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for improvements
   - `Docs:` for documentation changes
   - `Refactor:` for code refactoring

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Describe what changes you made and why

## Code Style Guidelines

### Python Code Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Keep functions focused and single-purpose
- Add docstrings to functions and classes
- Use type hints where appropriate

Example:
```python
def calculate_metrics(predictions: np.ndarray, 
                     ground_truth: np.ndarray, 
                     k: int = 20) -> Dict[str, float]:
    """
    Calculate evaluation metrics for recommendations.
    
    Args:
        predictions: Predicted item rankings for each user
        ground_truth: Actual user-item interactions
        k: Number of top recommendations to consider
        
    Returns:
        Dictionary containing metric names and values
    """
    # Implementation here
    pass
```

### Code Organization

- Keep related functionality together
- Use meaningful file and folder names
- Avoid circular dependencies
- Separate concerns (data loading, model definition, training, evaluation)

## Development Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Pre-commit Hooks (Optional)**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Testing Guidelines

- Test your changes with sample data
- Verify that training scripts work end-to-end
- Check that evaluation metrics are computed correctly
- Test with both CPU and GPU if possible

## Documentation

When adding new features:
- Update README.md with usage examples
- Add docstrings to new functions and classes
- Update relevant documentation files
- Include example commands or code snippets

## Areas for Contribution

We especially welcome contributions in these areas:

### New Features
- Additional recommendation models
- More evaluation metrics
- Data preprocessing utilities
- Hyperparameter tuning tools
- Visualization tools

### Improvements
- Performance optimization
- Better error handling
- Enhanced logging
- Code refactoring

### Documentation
- Tutorial notebooks
- Example use cases
- API documentation
- Performance benchmarks

### Testing
- Unit tests
- Integration tests
- Test data generators

## Questions?

If you have questions about contributing, feel free to:
- Open an issue on GitHub
- Contact the maintainers
- Ask in pull request discussions

## Code of Conduct

Please note that this project follows a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Recognition

Contributors will be acknowledged in:
- The project README
- Release notes
- Contributors list on GitHub

Thank you for contributing to making this project better!
