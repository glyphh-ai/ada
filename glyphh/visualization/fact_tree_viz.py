"""
Fact Tree Visualization

This module provides functions to visualize fact trees in various formats.
"""

from typing import Optional
from glyphh.fact_tree.builder import FactTree, FactNode


def visualize_fact_tree(
    fact_tree: FactTree,
    max_depth: Optional[int] = None,
    show_citations: bool = True,
    show_data_samples: bool = True,
    show_math: bool = True
) -> str:
    """
    Visualize a fact tree with customizable detail level.
    
    Args:
        fact_tree: FactTree to visualize
        max_depth: Maximum depth to display (None = unlimited)
        show_citations: Whether to show citations (default: True)
        show_data_samples: Whether to show data samples (default: True)
        show_math: Whether to show mathematical explanations (default: True)
    
    Returns:
        Formatted fact tree string
    
    Example:
        >>> viz = visualize_fact_tree(result.fact_tree, max_depth=3)
        >>> print(viz)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Fact Tree Visualization")
    lines.append("=" * 70)
    lines.append("")
    
    # Recursively build tree
    _build_tree_lines(
        fact_tree.root,
        lines,
        indent=0,
        max_depth=max_depth,
        current_depth=0,
        show_citations=show_citations,
        show_data_samples=show_data_samples,
        show_math=show_math
    )
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def _build_tree_lines(
    node: FactNode,
    lines: list,
    indent: int,
    max_depth: Optional[int],
    current_depth: int,
    show_citations: bool,
    show_data_samples: bool,
    show_math: bool
):
    """
    Recursively build tree lines for a fact node.
    
    Args:
        node: Current fact node
        lines: List to append lines to
        indent: Current indentation level
        max_depth: Maximum depth to display
        current_depth: Current depth in tree
        show_citations: Whether to show citations
        show_data_samples: Whether to show data samples
        show_math: Whether to show mathematical explanations
    """
    # Check depth limit
    if max_depth is not None and current_depth > max_depth:
        lines.append("  " * indent + "...")
        return
    
    # Node description and value
    prefix = "  " * indent
    if indent > 0:
        prefix = "  " * (indent - 1) + "+- "
    
    if node.value is not None:
        lines.append(f"{prefix}{node.description}: {node.value}")
    else:
        lines.append(f"{prefix}{node.description}")
    
    # Mathematical explanation
    if show_math and node.math_explanation:
        lines.append("  " * indent + f"|  Formula: {node.math_explanation}")
    
    # Data sample
    if show_data_samples and node.data_sample is not None:
        sample_str = str(node.data_sample)
        if len(sample_str) > 100:
            sample_str = sample_str[:100] + "..."
        lines.append("  " * indent + f"|  Sample: {sample_str}")
    
    # Citations
    if show_citations and node.citations:
        lines.append("  " * indent + "|  Citations:")
        for citation in node.citations:
            lines.append(
                "  " * indent + 
                f"|    - {citation.glyph_id}/{citation.component} "
                f"(v{citation.version}, {citation.timestamp})"
            )
    
    # Data context
    if node.data_context:
        lines.append("  " * indent + "|  Data Context:")
        for key, value in node.data_context.items():
            lines.append("  " * indent + f"|    - {key}: {value}")
    
    # Recursively process children
    for child in node.children:
        _build_tree_lines(
            child,
            lines,
            indent + 1,
            max_depth,
            current_depth + 1,
            show_citations,
            show_data_samples,
            show_math
        )


def print_fact_tree(
    fact_tree: FactTree,
    compact: bool = False
) -> str:
    """
    Print fact tree in a simple, readable format.
    
    Args:
        fact_tree: FactTree to print
        compact: Whether to use compact format (default: False)
    
    Returns:
        Formatted fact tree string
    
    Example:
        >>> print(print_fact_tree(result.fact_tree))
    """
    if compact:
        return _print_compact_tree(fact_tree)
    else:
        return fact_tree.to_text()


def _print_compact_tree(fact_tree: FactTree) -> str:
    """
    Print fact tree in compact format (no citations, samples, or math).
    
    Args:
        fact_tree: FactTree to print
    
    Returns:
        Compact fact tree string
    """
    lines = []
    _build_compact_lines(fact_tree.root, lines, indent=0)
    return "\n".join(lines)


def _build_compact_lines(node: FactNode, lines: list, indent: int):
    """
    Recursively build compact tree lines.
    
    Args:
        node: Current fact node
        lines: List to append lines to
        indent: Current indentation level
    """
    prefix = "  " * indent
    if indent > 0:
        prefix = "  " * (indent - 1) + "+- "
    
    if node.value is not None:
        lines.append(f"{prefix}{node.description}: {node.value}")
    else:
        lines.append(f"{prefix}{node.description}")
    
    for child in node.children:
        _build_compact_lines(child, lines, indent + 1)


def summarize_fact_tree(fact_tree: FactTree) -> str:
    """
    Create a summary of the fact tree.
    
    Args:
        fact_tree: FactTree to summarize
    
    Returns:
        Summary string
    
    Example:
        >>> summary = summarize_fact_tree(result.fact_tree)
        >>> print(summary)
    """
    # Count nodes, citations, and depth
    node_count = _count_nodes(fact_tree.root)
    citation_count = _count_citations(fact_tree.root)
    max_depth = _compute_max_depth(fact_tree.root, 0)
    
    lines = []
    lines.append("=" * 70)
    lines.append("Fact Tree Summary")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Root: {fact_tree.root.description}")
    lines.append(f"Total nodes: {node_count}")
    lines.append(f"Total citations: {citation_count}")
    lines.append(f"Maximum depth: {max_depth}")
    lines.append("")
    
    # Top-level facts
    lines.append("Top-level facts:")
    for child in fact_tree.root.children:
        if child.value is not None:
            lines.append(f"  - {child.description}: {child.value}")
        else:
            lines.append(f"  - {child.description}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def _count_nodes(node: FactNode) -> int:
    """Count total nodes in tree."""
    count = 1
    for child in node.children:
        count += _count_nodes(child)
    return count


def _count_citations(node: FactNode) -> int:
    """Count total citations in tree."""
    count = len(node.citations)
    for child in node.children:
        count += _count_citations(child)
    return count


def _compute_max_depth(node: FactNode, current_depth: int) -> int:
    """Compute maximum depth of tree."""
    if not node.children:
        return current_depth
    
    max_child_depth = current_depth
    for child in node.children:
        child_depth = _compute_max_depth(child, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)
    
    return max_child_depth


def export_fact_tree_json(fact_tree: FactTree) -> str:
    """
    Export fact tree as JSON string.
    
    Args:
        fact_tree: FactTree to export
    
    Returns:
        JSON string
    
    Example:
        >>> json_str = export_fact_tree_json(result.fact_tree)
        >>> with open("fact_tree.json", "w") as f:
        ...     f.write(json_str)
    """
    import json
    return json.dumps(fact_tree.to_json(), indent=2)


def export_fact_tree_markdown(fact_tree: FactTree) -> str:
    """
    Export fact tree as Markdown.
    
    Args:
        fact_tree: FactTree to export
    
    Returns:
        Markdown string
    
    Example:
        >>> md = export_fact_tree_markdown(result.fact_tree)
        >>> with open("fact_tree.md", "w") as f:
        ...     f.write(md)
    """
    lines = []
    lines.append(f"# {fact_tree.root.description}")
    lines.append("")
    
    _build_markdown_lines(fact_tree.root, lines, level=2)
    
    return "\n".join(lines)


def _build_markdown_lines(node: FactNode, lines: list, level: int):
    """
    Recursively build markdown lines.
    
    Args:
        node: Current fact node
        lines: List to append lines to
        level: Current heading level
    """
    for child in node.children:
        # Heading
        heading = "#" * level
        if child.value is not None:
            lines.append(f"{heading} {child.description}: {child.value}")
        else:
            lines.append(f"{heading} {child.description}")
        lines.append("")
        
        # Math explanation
        if child.math_explanation:
            lines.append(f"**Formula:** `{child.math_explanation}`")
            lines.append("")
        
        # Data sample
        if child.data_sample is not None:
            lines.append(f"**Sample:** {child.data_sample}")
            lines.append("")
        
        # Citations
        if child.citations:
            lines.append("**Citations:**")
            for citation in child.citations:
                lines.append(
                    f"- {citation.glyph_id}/{citation.component} "
                    f"(v{citation.version}, {citation.timestamp})"
                )
            lines.append("")
        
        # Data context
        if child.data_context:
            lines.append("**Data Context:**")
            for key, value in child.data_context.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        
        # Recurse
        _build_markdown_lines(child, lines, level + 1)
