"""
Package commands for model packaging and validation
"""

import click
import json
from pathlib import Path
from datetime import datetime
from ..core import EncoderConfig, Concept
from ..encoder import Encoder
from ..model import GlyphhModel


@click.group()
def package():
    """Package commands for model deployment"""
    pass


@package.command()
@click.option('--model', type=click.Path(exists=True), default='model.json', help='Model file path')
@click.option('--readme', type=click.Path(), default='readme.md', help='README file path')
@click.option('--output', type=click.Path(), required=True, help='Output .glyphh file path')
@click.option('--version', default='1.0.0', help='Model version (semantic versioning)')
def create(model: str, readme: str, output: str, version: str):
    """
    Create a packaged model file (.glyphh)
    
    Example:
        glyphh package create --output my_model.glyphh
        glyphh package create --model model.json --readme readme.md --output model.glyphh --version 1.2.0
    """
    try:
        # Load model
        model_path = Path(model)
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Load readme if exists
        readme_content = None
        readme_path = Path(readme)
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            click.echo(f"[OK] Loaded readme: {readme_path}")
        else:
            click.echo(f"Warning: No readme found at {readme_path}", err=True)
        
        # Create encoder
        config_data = model_data['config']
        config = EncoderConfig(
            dimension=config_data['dimension'],
            seed=config_data['seed'],
            similarity_weight=config_data.get('similarity_weight', 1.0),
            security_weight=config_data.get('security_weight', 1.0),
        )
        encoder = Encoder(config)
        
        # Encode all concepts
        glyphs = []
        concepts = model_data.get('concepts', [])
        
        if not concepts:
            click.echo("Warning: Model has no concepts", err=True)
        
        click.echo(f"Encoding {len(concepts)} concepts...")
        for concept_data in concepts:
            concept = Concept(
                name=concept_data['name'],
                attributes=concept_data['attributes']
            )
            glyph = encoder.encode(concept)
            glyphs.append(glyph)
            click.echo(f"  [OK] Encoded: {concept.name}")
        
        # Create packaged model
        packaged_model = GlyphhModel(
            name=model_data['name'],
            version=version,
            encoder_config=config,
            glyphs=glyphs,
            custom_encoders={},
            readme=readme_content,
            metadata={
                'created_at': datetime.now().isoformat(),
                'source_file': str(model_path)
            }
        )
        
        # Validate model
        errors = packaged_model.validate_completeness()
        if errors:
            click.echo("Error: Model validation failed:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            raise click.Abort()
        
        # Save to file
        output_path = Path(output)
        packaged_model.to_file(str(output_path))
        
        click.echo()
        click.echo(f"[OK] Created package: {output_path}")
        click.echo(f"  Model: {packaged_model.name}")
        click.echo(f"  Version: {packaged_model.version}")
        click.echo(f"  Glyphs: {len(packaged_model.glyphs)}")
        click.echo(f"  Dimension: {config.dimension}")
        
    except FileNotFoundError:
        click.echo(f"Error: Model file not found: {model}", err=True)
        click.echo("  Run 'build init' first to create a model", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error creating package: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise click.Abort()


@package.command()
@click.argument('package_file', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def validate(package_file: str, format: str):
    """
    Validate a packaged model file
    
    Example:
        glyphh package validate model.glyphh
        glyphh package validate model.glyphh --format json
    """
    try:
        # Load package
        package_path = Path(package_file)
        model = GlyphhModel.from_file(str(package_path))
        
        # Validate
        errors = model.validate_completeness()
        
        if format == 'json':
            # JSON output for scripting
            output = {
                'valid': len(errors) == 0,
                'errors': errors,
                'model_name': model.name,
                'version': model.version,
                'glyph_count': len(model.glyphs)
            }
            click.echo(json.dumps(output, indent=2))
        else:
            # Human-readable text output
            click.echo(f"Package Validation Results")
            click.echo(f"=" * 50)
            click.echo(f"File: {package_path}")
            click.echo(f"Model: {model.name}")
            click.echo(f"Version: {model.version}")
            click.echo(f"Glyphs: {len(model.glyphs)}")
            click.echo()
            
            if errors:
                click.echo("Validation FAILED")
                click.echo()
                click.echo("Errors:")
                for error in errors:
                    click.echo(f"  - {error}")
            else:
                click.echo("[OK] Validation PASSED")
                click.echo("  Model is ready for deployment")
        
    except FileNotFoundError:
        click.echo(f"Error: Package file not found: {package_file}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error validating package: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise click.Abort()


@package.command()
@click.argument('package_file', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def info(package_file: str, format: str):
    """
    Display metadata about a packaged model
    
    Example:
        glyphh package info model.glyphh
        glyphh package info model.glyphh --format json
    """
    try:
        # Load package
        package_path = Path(package_file)
        model = GlyphhModel.from_file(str(package_path))
        
        if format == 'json':
            # JSON output for scripting
            output = {
                'name': model.name,
                'version': model.version,
                'glyph_count': len(model.glyphs),
                'dimension': model.encoder_config.dimension,
                'seed': model.encoder_config.seed,
                'similarity_weight': model.encoder_config.similarity_weight,
                'security_weight': model.encoder_config.security_weight,
                'created_at': model.metadata.get('created_at'),
                'metadata': model.metadata
            }
            click.echo(json.dumps(output, indent=2))
        else:
            # Human-readable text output
            click.echo(f"Package Information")
            click.echo(f"=" * 50)
            click.echo(f"File: {package_path}")
            click.echo(f"Name: {model.name}")
            click.echo(f"Version: {model.version}")
            click.echo(f"Created: {model.metadata.get('created_at', 'Unknown')}")
            click.echo()
            click.echo(f"Configuration:")
            click.echo(f"  Dimension: {model.encoder_config.dimension}")
            click.echo(f"  Seed: {model.encoder_config.seed}")
            click.echo(f"  Layers: {len(model.encoder_config.layers)}")
            click.echo()
            click.echo(f"Content:")
            click.echo(f"  Glyphs: {len(model.glyphs)}")
            click.echo(f"  Custom encoders: {len(model.custom_encoders)}")
            click.echo(f"  Readme: {'Yes' if model.readme else 'No'}")
            
            if model.glyphs:
                click.echo()
                click.echo(f"Glyphs (first 5):")
                for i, glyph in enumerate(model.glyphs[:5], 1):
                    click.echo(f"  {i}. {glyph.name}")
                
                if len(model.glyphs) > 5:
                    click.echo(f"  ... and {len(model.glyphs) - 5} more")
        
    except FileNotFoundError:
        click.echo(f"Error: Package file not found: {package_file}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error reading package info: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise click.Abort()
