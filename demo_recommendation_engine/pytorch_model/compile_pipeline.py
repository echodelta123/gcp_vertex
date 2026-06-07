"""Compile the recommendation pipeline to a Vertex AI pipeline spec JSON.

Usage:
    python -m demo_recommendation_engine.pytorch_model.compile_pipeline
"""

from __future__ import annotations

from kfp import compiler

from .pipeline import recommendation_pipeline


def main() -> None:
    output_path = "recommendation_engine_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=recommendation_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to: {output_path}")


if __name__ == "__main__":
    main()
