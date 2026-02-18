"""
Pytest configuration for property-based tests.

This module configures Hypothesis settings for all property tests.
"""

from hypothesis import settings, Verbosity

# Register a profile for property tests
settings.register_profile(
    "glyphh_property_tests",
    max_examples=100,
    deadline=None,  # No deadline for property tests (some may be slow)
    verbosity=Verbosity.normal,
    print_blob=True,  # Print failing examples
)

# Load the profile
settings.load_profile("glyphh_property_tests")
