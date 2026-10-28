## Getting Started

### Prerequisites

- [Python](https://www.python.org/) 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and environment manager

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd LiquifyTool
   ```

2. Install dependencies and set up the virtual environment:

   ```bash
   uv sync
   ```

   This automatically creates a virtual environment and installs the required libraries, including NumPy, Matplotlib, and svgpathtools.

### Running the Project

Launch the interactive Bézier editor:

```bash
uv run python main.py
```

This opens a Matplotlib window displaying the Bézier curves loaded from `example.svg`.

### Using the Editor

- **Drag control points** to modify the shape of a Bézier curve.
- **Drag shared endpoints** to move connected segments together.
- **Move endpoints** to translate their adjacent control handles along with them.

### Project Structure

```text
LiquifyTool/
├── main.py                # Application entry point
├── interactive_bezier.py  # Interactive Bézier curve editor
├── read_svg_cubics.py     # SVG to cubic Bézier converter
├── example.svg            # Example SVG input
├── pyproject.toml         # Project dependencies
└── uv.lock                # Locked dependency versions
```

### Development

To install additional dependencies:

```bash
uv add <package-name>
```

To synchronize dependencies after pulling changes:

```bash
uv sync
```