"""
Test commands for similarity testing and encoding validation
"""

import click
import json
from pathlib import Path
from typing import Optional
from ..core import EncoderConfig, Concept
from ..encoder import Encoder
from ..similarity import SimilarityCalculator


@click.group()
def test():
    """Test commands for validation and similarity testing"""
    pass


@test.command()
@click.argument('glyph1')
@click.argument('glyph2')
@click.option('--model', type=click.Path(exists=True), default='model.json', help='Model file path')
@click.option('--edge-type', default='neural_cortex', help='Edge type for similarity (neural_cortex, neural_layer, etc.)')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--show-fact-tree', is_flag=True, help='Show detailed fact tree explanation')
def similarity(glyph1: str, glyph2: str, model: str, edge_type: str, format: str, show_fact_tree: bool):
    """
    Test similarity between two glyphs
    
    Example:
        glyphh test similarity "red car" "blue car"
        glyphh test similarity glyph1 glyph2 --edge-type neural_layer
        glyphh test similarity glyph1 glyph2 --format json
    """
    try:
        # Load model
        model_path = Path(model)
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Find concepts
        concepts = model_data.get('concepts', [])
        concept1_data = next((c for c in concepts if c['name'] == glyph1), None)
        concept2_data = next((c for c in concepts if c['name'] == glyph2), None)
        
        if not concept1_data:
            click.echo(f"Error: Concept '{glyph1}' not found in model", err=True)
            raise click.Abort()
        
        if not concept2_data:
            click.echo(f"Error: Concept '{glyph2}' not found in model", err=True)
            raise click.Abort()
        
        # Create encoder
        config_data = model_data['config']
        config = EncoderConfig(
            dimension=config_data['dimension'],
            seed=config_data['seed'],
            similarity_weight=config_data.get('similarity_weight', 1.0),
            security_weight=config_data.get('security_weight', 1.0),
        )
        encoder = Encoder(config)
        
        # Encode concepts
        concept1 = Concept(
            name=concept1_data['name'],
            attributes=concept1_data['attributes']
        )
        concept2 = Concept(
            name=concept2_data['name'],
            attributes=concept2_data['attributes']
        )
        
        glyph1_obj = encoder.encode(concept1)
        glyph2_obj = encoder.encode(concept2)
        
        # Compute similarity
        calculator = SimilarityCalculator()
        result = calculator.compute_similarity(glyph1_obj, glyph2_obj, edge_type)
        
        if format == 'json':
            # JSON output for scripting
            output = {
                'glyph1': glyph1,
                'glyph2': glyph2,
                'edge_type': edge_type,
                'similarity_score': float(result.score),
                'visible': result.visible
            }
            if show_fact_tree:
                output['fact_tree'] = result.fact_tree.to_json()
            click.echo(json.dumps(output, indent=2))
        else:
            # Human-readable text output
            click.echo(f"Similarity Test Results")
            click.echo(f"=" * 50)
            click.echo(f"Glyph 1: {glyph1}")
            click.echo(f"Glyph 2: {glyph2}")
            click.echo(f"Edge Type: {edge_type}")
            click.echo(f"Similarity Score: {result.score:.4f}")
            click.echo(f"Visible: {'Yes' if result.visible else 'No'}")
            
            if show_fact_tree:
                click.echo()
                click.echo("Fact Tree:")
                click.echo("-" * 50)
                click.echo(result.fact_tree.to_text())
        
    except FileNotFoundError:
        click.echo(f"Error: Model file not found: {model}", err=True)
        click.echo("  Run 'build init' first to create a model", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error computing similarity: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise click.Abort()


@test.command()
@click.option('--concept', required=True, help='Concept name')
@click.option('--attributes', required=True, help='JSON string of attributes')
@click.option('--model', type=click.Path(exists=True), default='model.json', help='Model file path')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--show-vector', is_flag=True, help='Show vector sample')
def encode(concept: str, attributes: str, model: str, format: str, show_vector: bool):
    """
    Test encoding of a concept
    
    Example:
        glyphh test encode --concept "blue car" --attributes '{"type":"car","color":"blue"}'
    """
    try:
        # Load model
        model_path = Path(model)
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Create encoder
        config_data = model_data['config']
        config = EncoderConfig(
            dimension=config_data['dimension'],
            seed=config_data['seed'],
            similarity_weight=config_data.get('similarity_weight', 1.0),
            security_weight=config_data.get('security_weight', 1.0),
        )
        encoder = Encoder(config)
        
        # Parse attributes
        try:
            attrs = json.loads(attributes)
        except json.JSONDecodeError as e:
            click.echo(f"Error: Invalid JSON in attributes: {e}", err=True)
            raise click.Abort()
        
        # Encode concept
        concept_obj = Concept(name=concept, attributes=attrs)
        glyph = encoder.encode(concept_obj)
        
        if format == 'json':
            # JSON output for scripting
            output = {
                'concept': concept,
                'attributes': attrs,
                'space_id': glyph.space_id,
                'dimension': len(glyph.global_cortex.data),
                'num_layers': len(glyph.layers),
                'identifier': glyph.identifier
            }
            if show_vector:
                output['vector_sample'] = glyph.global_cortex.data[:20].tolist()
            click.echo(json.dumps(output, indent=2))
        else:
            # Human-readable text output
            click.echo(f"Encoding Test Results")
            click.echo(f"=" * 50)
            click.echo(f"Concept: {concept}")
            click.echo(f"Attributes: {attrs}")
            click.echo(f"Space ID: {glyph.space_id}")
            click.echo(f"Dimension: {len(glyph.global_cortex.data)}")
            click.echo(f"Layers: {len(glyph.layers)}")
            click.echo(f"Identifier: {glyph.identifier}")
            
            if show_vector:
                click.echo()
                click.echo("Vector Sample (first 20 dimensions):")
                click.echo(glyph.global_cortex.data[:20].tolist())
        
    except FileNotFoundError:
        click.echo(f"Error: Model file not found: {model}", err=True)
        click.echo("  Run 'build init' first to create a model", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error encoding concept: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise click.Abort()
