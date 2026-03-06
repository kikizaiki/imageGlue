"""FastAPI application."""
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.logging import setup_logging
from app.pipelines.render_pipeline import RenderPipeline

setup_logging()

app = FastAPI(
    title="imageGlue API",
    description="Dog poster rendering service",
    version="1.0.0",
)

pipeline = RenderPipeline()


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    template_id: str = "dog_cosmonaut_v1",
    debug: bool = False,
):
    """
    Render dog image into template.

    Args:
        image: Input image file
        template_id: Template identifier
        debug: Enable debug output

    Returns:
        Rendered image file
    """
    try:
        # Save uploaded file temporarily
        from tempfile import NamedTemporaryFile
        import shutil

        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            shutil.copyfileobj(image.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Render
            final_image, metadata = pipeline.render(
                image_path=tmp_path,
                template_id=template_id,
                debug=debug,
            )

            # Save to output
            output_path = metadata["output_path"]
            return FileResponse(
                output_path,
                media_type="image/png",
                filename="result.png",
            )

        finally:
            # Cleanup
            from pathlib import Path
            Path(tmp_path).unlink()

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}
