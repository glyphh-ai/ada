"""
Demo reel entry point — delegates to individual reels.

Kept for backward compatibility. The actual reels live in
reel_product.py and reel_churn.py.
"""

from .reel_product import show_reel_product

# Default reel is the product demo
show_demo_reel = show_reel_product
