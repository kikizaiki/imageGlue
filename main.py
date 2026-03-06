"""Main entry point for CLI."""
import argparse
import sys
from pathlib import Path

from app.core.logging import setup_logging
from app.pipelines.render_pipeline import RenderPipeline

setup_logging()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="imageGlue - Dog poster renderer")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Input image path",
    )
    parser.add_argument(
        "--template",
        "-t",
        required=True,
        type=str,
        help="Template ID",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output image path (default: runs/output/<job_id>/result.png)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    try:
        pipeline = RenderPipeline()
        final_image, metadata = pipeline.render(
            image_path=args.input,
            template_id=args.template,
            debug=args.debug,
        )

        # Save output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_image.save(output_path, quality=95)
            print(f"✅ Result saved: {output_path}")
        else:
            print(f"✅ Result saved: {metadata['output_path']}")

        print(f"📊 Quality score: {metadata.get('quality', {}).get('overall', 0):.2f}")
        print(f"⏱️  Total time: {sum(metadata.get('timings', {}).values()):.2f}s")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
