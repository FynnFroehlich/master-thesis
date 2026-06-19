# Thesis Diagrams

This folder contains architecture and class diagrams for the renewable energy project simulation framework.

## Files

| File | Description |
|------|-------------|
| `architecture_diagram.md` | Comprehensive documentation with all diagrams in Mermaid format |
| `flowchart_standalone.mmd` | High-level system architecture flowchart |
| `class_diagram_standalone.mmd` | Class/dataclass relationship diagram |

## Rendering Options

### Option 1: Mermaid Live Editor (Easiest)

1. Go to https://mermaid.live/
2. Copy-paste the content from any `.mmd` file
3. Export as PNG, SVG, or PDF

### Option 2: VS Code Extension

1. Install "Mermaid Preview" or "Markdown Preview Mermaid Support" extension
2. Open the `.md` or `.mmd` file
3. Use preview pane to view diagrams

### Option 3: Command Line (mermaid-cli)

```bash
# Install mermaid-cli globally
npm install -g @mermaid-js/mermaid-cli

# Export to PNG (recommended for thesis)
mmdc -i flowchart_standalone.mmd -o flowchart.png -w 1600 -b white

# Export to SVG (scalable)
mmdc -i flowchart_standalone.mmd -o flowchart.svg

# Export to PDF
mmdc -i flowchart_standalone.mmd -o flowchart.pdf

# Export class diagram
mmdc -i class_diagram_standalone.mmd -o class_diagram.png -w 1400 -b white
```

### Option 4: Python (for integration with notebooks)

```python
# Install: pip install mermaid-py
from mermaid import Mermaid

with open('flowchart_standalone.mmd', 'r') as f:
    content = f.read()

# Render in Jupyter
Mermaid(content)
```

## For LaTeX Thesis

If including in LaTeX, export to PDF or PNG and use:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{diagrams/flowchart.png}
    \caption{High-level architecture of the renewable energy simulation framework}
    \label{fig:architecture}
\end{figure}
```

## Diagram Overview

### System Architecture
Shows the modular structure:
- **Data Layer**: Raw data loading and preprocessing
- **Configuration**: Techno-economic parameters
- **Simulation Engine**: Rule-based and LP-optimized dispatch
- **Analysis Modules**: Different scenario evaluations
- **Economics**: NPV, IRR, cash flow calculations

### Class Diagram
Shows the main data structures:
- `Config`: Global configuration parameters
- `Scenario`: Project scenario definition
- Result classes for different analysis types:
  - `OversizingResult` (stand-alone PV/Wind)
  - `HybridResult` (PV+Wind)
  - `HybridBessResult` (PV+Wind+BESS)
  - `PvBessResult` (PV+BESS)
  - `PvBessNpvResult` (detailed economics)

## Key Parameters

| Symbol | Description | Default |
|--------|-------------|---------|
| α (alpha) | Oversizing factor = P_installed / P_POI | varies |
| α_tot | Total hybrid overplanting | varies |
| s_wind | Wind share = P_wind / (P_pv + P_wind) | 0-1 |
| γ (gamma) | Battery power ratio = P_BESS / P_POI | varies |
| POI | Point of Interconnection capacity | 10 MW |
