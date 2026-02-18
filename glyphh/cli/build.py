"""
Build commands for model initialization and concept management
"""

import click
import json
from pathlib import Path
from typing import Optional
from ..core import EncoderConfig, Concept
from ..encoder import Encoder
from ..model import GlyphhModel


# Global state for current model (in-memory for now)
_current_model: Optional[dict] = None


@click.group()
def build():
    """Build commands for model creation and management"""
    pass


@build.command()
@click.option('--name', required=True, help='Model name')
@click.option('--dimension', type=int, required=True, help='Vector dimension (e.g., 10000)')
@click.option('--seed', type=int, default=42, help='Random seed for deterministic encoding')
@click.option('--output', type=click.Path(), default='model.json', help='Output file path')
def init(name: str, dimension: int, seed: int, output: str):
    """
    Initialize a new model
    
    Example:
        glyphh build init --name my_model --dimension 10000 --seed 42
    """
    global _current_model
    
    try:
        # Create encoder configuration (new explicit structure)
        config = EncoderConfig(
            dimension=dimension,
            seed=seed,
        )
        
        # Initialize model structure
        _current_model = {
            'name': name,
            'config': {
                'dimension': dimension,
                'seed': seed,
                'similarity_weight': 1.0,
                'security_weight': 1.0,
                'apply_weights_during_encoding': False,
                'layers': []
            },
            'concepts': []
        }
        
        # Save to file
        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump(_current_model, f, indent=2)
        
        click.echo(f"[OK] Initialized model '{name}'")
        click.echo(f"  Dimension: {dimension}")
        click.echo(f"  Seed: {seed}")
        click.echo(f"  Saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error initializing model: {e}", err=True)
        raise click.Abort()


@build.command()
@click.option('--concept', required=True, help='Concept name (e.g., "red car")')
@click.option('--attributes', required=True, help='JSON string of attributes (e.g., \'{"type":"car","color":"red"}\')')
@click.option('--model', type=click.Path(exists=True), default='model.json', help='Model file path')
@click.option('--output', type=click.Path(), help='Output file path (defaults to input model path)')
def add(concept: str, attributes: str, model: str, output: Optional[str]):
    """
    Add a concept to the model
    
    Example:
        glyphh build add --concept "red car" --attributes '{"type":"car","color":"red"}'
    """
    try:
        # Load model
        model_path = Path(model)
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Parse attributes
        try:
            attrs = json.loads(attributes)
        except json.JSONDecodeError as e:
            click.echo(f"Error: Invalid JSON in attributes: {e}", err=True)
            raise click.Abort()
        
        # Add concept to model
        concept_data = {
            'name': concept,
            'attributes': attrs
        }
        model_data['concepts'].append(concept_data)
        
        # Save updated model
        output_path = Path(output) if output else model_path
        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        click.echo(f"[OK] Added concept '{concept}' to model")
        click.echo(f"  Attributes: {attrs}")
        click.echo(f"  Total concepts: {len(model_data['concepts'])}")
        
    except FileNotFoundError:
        click.echo(f"Error: Model file not found: {model}", err=True)
        click.echo("  Run 'build init' first to create a model", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error adding concept: {e}", err=True)
        raise click.Abort()


@build.command()
@click.option('--model', type=click.Path(exists=True), default='model.json', help='Model file path')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def list(model: str, format: str):
    """
    List all concepts in the model
    
    Example:
        glyphh build list
        glyphh build list --format json
    """
    try:
        # Load model
        model_path = Path(model)
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        concepts = model_data.get('concepts', [])
        
        if format == 'json':
            # JSON output for scripting
            output = {
                'model_name': model_data['name'],
                'total_concepts': len(concepts),
                'concepts': concepts
            }
            click.echo(json.dumps(output, indent=2))
        else:
            # Human-readable text output
            click.echo(f"Model: {model_data['name']}")
            click.echo(f"Total concepts: {len(concepts)}")
            click.echo()
            
            if concepts:
                click.echo("Concepts:")
                for i, concept in enumerate(concepts, 1):
                    click.echo(f"  {i}. {concept['name']}")
                    for key, value in concept['attributes'].items():
                        click.echo(f"     {key}: {value}")
            else:
                click.echo("No concepts added yet.")
                click.echo("Use 'build add' to add concepts.")
        
    except FileNotFoundError:
        click.echo(f"Error: Model file not found: {model}", err=True)
        click.echo("  Run 'build init' first to create a model", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error listing concepts: {e}", err=True)
        raise click.Abort()
