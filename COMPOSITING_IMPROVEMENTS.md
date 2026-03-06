# Compositing Quality Improvements

This document describes the refactoring to support proper masked template compositing.

## Summary of Changes

The pipeline has been refactored to support:
- Real background removal using rembg
- 4-directional crop expansion
- Template mask support (visor clipping)
- Edge cleanup and alpha refinement
- Color integration for better blending
- Enhanced debug artifacts

## New Features

### 1. Real Background Removal

**Provider**: `RembgBackgroundRemovalProvider`
- Uses rembg library with u2net model (configurable)
- Properly extracts subject with alpha channel
- Set `BGREMOVAL_BACKEND=rembg` in `.env`

**Installation**:
```bash
pip install rembg[new]
```

### 2. 4-Directional Crop Expansion

**Function**: `expand_bbox_directional()`

Instead of uniform expansion, you can now specify:
- `expand_left`: Expansion factor for left side
- `expand_right`: Expansion factor for right side
- `expand_top`: Expansion factor for top side
- `expand_bottom`: Expansion factor for bottom side
- `crop_vertical_shift`: Shift crop downward (in pixels)

**Example config**:
```json
"crop_expansion": {
  "expand_left": 0.4,
  "expand_right": 0.4,
  "expand_top": 0.3,
  "expand_bottom": 0.6
},
"crop_vertical_shift": 30
```

This captures head + ears + neck + upper chest, not just the snout.

### 3. Template Mask Support

**New config field**: `visor_mask`

The subject is now clipped by a mask before final compositing:
- Mask file: `visor_mask.png` (grayscale or RGBA)
- Applied after subject placement
- Multiplies alpha channels for proper clipping

**Config**:
```json
"visor_mask": {
  "path": "visor_mask.png"
}
```

### 4. Edge Cleanup

**Module**: `app/services/composer/edge_cleanup.py`

Features:
- **Feather alpha**: Softens edges with Gaussian blur
- **Erode/Dilate**: Shrink or expand alpha channel
- **Remove white halo**: Makes near-white pixels transparent

**Config**:
```json
"compositing": {
  "feather_alpha": true,
  "edge_cleanup": true,
  "alpha_feather_px": 3,
  "alpha_erode_px": 0,
  "remove_halo": true,
  "halo_threshold": 240
}
```

### 5. Color Integration

**Module**: `app/services/composer/color_integration.py`

Adjusts subject colors to match template:
- Contrast adjustment
- Brightness correction
- Color tinting (RGB multipliers)

**Config**:
```json
"color_match": {
  "enabled": true,
  "contrast": 1.05,
  "brightness": 0.98,
  "tint_rgb": [0.98, 0.99, 1.02]
}
```

### 6. Enhanced Placement

**New parameters**:
- `scale_multiplier`: Additional scale factor (default: 1.0)
- Improved `vertical_bias` and `horizontal_bias` support

**Config**:
```json
"placement": {
  "scale_multiplier": 1.2,
  "vertical_bias": 0.15,
  "horizontal_bias": 0.0
}
```

### 7. Debug Artifacts

The pipeline now saves:
- `original.png` - Original uploaded image
- `detection_overlay.png` - Detection visualization
- `expanded_crop.png` - Crop after expansion
- `subject_rgba.png` - Subject after background removal
- `subject_placed.png` - Subject placed on canvas (before mask)
- `subject_masked_by_visor.png` - Subject after mask application
- `final.png` - Final composed image
- `metadata.json` - Full job metadata

## Updated Files

### Core Changes

1. **`app/services/bgremove/rembg_provider.py`** (NEW)
   - Rembg background removal provider

2. **`app/services/composer/geometry.py`**
   - Added `expand_bbox_directional()`
   - Added `scale_multiplier` to `compute_placement()`

3. **`app/services/composer/edge_cleanup.py`** (NEW)
   - Alpha edge cleanup utilities

4. **`app/services/composer/color_integration.py`** (NEW)
   - Color matching utilities

5. **`app/services/composer/sandwich.py`**
   - Mask support
   - Edge cleanup integration
   - Color integration
   - Returns debug images

6. **`app/services/jobs/orchestrator.py`**
   - Uses directional crop expansion
   - Saves all debug artifacts
   - Uses rembg provider

7. **`app/core/storage.py`**
   - New methods: `save_expanded_crop()`, `save_subject_rgba()`, `save_debug_image()`

8. **`app/core/config.py`**
   - Added `BGREMOVAL_REMBG_MODEL` setting

9. **`app/services/templates/loader.py`**
   - Resolves visor mask paths

10. **`templates/astronaut_dog/config.json`**
    - Updated with all new configuration options

## Template Configuration

### Required Files

For the astronaut_dog template, you need:
- `background.png` - Background layer
- `foreground.png` - Foreground overlay (RGBA)
- `visor_mask.png` - **NEW**: Mask for clipping subject (grayscale or RGBA)

### Creating visor_mask.png

The visor mask should be:
- Same size as canvas (1024x1024)
- Grayscale or RGBA
- White = visible area, Black = clipped area
- Defines where the dog's head should appear in the helmet

You can create it by:
1. Using the foreground.png as a base
2. Creating a mask where the visor opening is white
3. Everything else is black/transparent

## Usage

### 1. Install Dependencies

```bash
pip install rembg[new] scipy numpy
```

### 2. Configure

Update `.env`:
```
BGREMOVAL_BACKEND=rembg
BGREMOVAL_REMBG_MODEL=u2net
```

### 3. Create Visor Mask

Create `templates/astronaut_dog/visor_mask.png`:
- White area = where dog head appears
- Black area = clipped

### 4. Test

```bash
python scripts/test_api.py /path/to/dog_image.jpg
```

## Pipeline Flow

1. **Ingest** → Save original
2. **Detection** → Find dog head
3. **Crop Planning** → Expand 4-directionally, shift down
4. **Background Removal** → Extract with rembg
5. **Subject Alignment** → Scale and place with multiplier
6. **Compositing**:
   - Apply edge cleanup
   - Apply color integration
   - Place subject
   - Apply visor mask
   - Composite: background → masked subject → foreground
7. **Delivery** → Save final + debug artifacts

## Backward Compatibility

- Old configs with `crop_expansion: 1.3` (number) still work
- New configs use `crop_expansion: { expand_left: 0.4, ... }` (object)
- Mask is optional - templates without mask work as before
- Edge cleanup and color match are optional

## Performance Notes

- rembg first run downloads model (~176MB)
- Subsequent runs are faster
- Edge cleanup with scipy is optional (graceful fallback)
- Color integration is lightweight

## Troubleshooting

### rembg not working
- Install: `pip install rembg[new]`
- Check model download (first run takes time)
- Try different model: `BGREMOVAL_REMBG_MODEL=u2net_human_seg`

### Mask not applying
- Check visor_mask.png exists
- Verify path in config.json
- Check mask is grayscale or RGBA
- Check logs for warnings

### Edge cleanup not working
- Install scipy: `pip install scipy`
- Or disable in config: `"edge_cleanup": false`

## Next Steps

1. Create visor_mask.png for your template
2. Tune crop_expansion values for your use case
3. Adjust color_match settings for better integration
4. Fine-tune scale_multiplier and biases
