#!/usr/bin/env python3
"""
Prompt Refinement Engine

Template-based synthesis for generating perfect image-text captions:
- Combines scene_type + layout_description + object_composition + appearance_details
- Verification by comparing analysis passes
- Quality scoring for synthetic-real alignment
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class PromptComponents:
    """Components for building a refined prompt."""
    scene_type: str
    layout_description: str
    objects_with_positions: str
    appearance_details: str
    lighting_conditions: str


class PromptRefinementEngine:
    """
    Engine for generating and refining prompts/captions.
    
    Uses template-based synthesis to create high-quality prompts
    for SD generation and refined captions for training.
    """
    
    # Base template for SD generation prompts
    SD_PROMPT_TEMPLATE = (
        "High-resolution aerial satellite view of {scene} area. "
        "Layout: {layout}. "
        "Objects: {objects}. "
        "Visual style: {appearance}, {lighting}. "
        "Photorealistic remote sensing imagery, sharp details, natural colors."
    )
    
    # Template for refined training captions
    CAPTION_TEMPLATE = (
        "{scene_desc} containing {objects}. {spatial_context}. {visual_details}."
    )
    
    # Scene type mappings for richer descriptions
    SCENE_EXPANSIONS = {
        "urban": "urban metropolitan",
        "residential": "residential neighborhood",
        "commercial": "commercial district",
        "industrial": "industrial complex",
        "agricultural": "agricultural farmland",
        "forest": "forested woodland",
        "water": "aquatic waterfront",
        "rural": "rural countryside",
        "suburban": "suburban residential",
        "airport": "airport facility",
        "port": "maritime port",
        "parking": "parking facility",
        "stadium": "sports stadium",
        "park": "recreational park"
    }
    
    # Object count words
    COUNT_WORDS = {
        1: "a single",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "several",
        8: "several",
        9: "numerous",
        10: "many"
    }
    
    def __init__(self):
        """Initialize the prompt refinement engine."""
        pass
    
    def generate_sd_prompt(
        self,
        scene_type: str,
        object_inventory: Dict[str, int],
        layout_description: str,
        appearance_details: str,
        lighting_conditions: str,
        spatial_relationships: Optional[List[str]] = None
    ) -> str:
        """
        Generate an optimized prompt for Stable Diffusion.
        
        Args:
            scene_type: Type of scene (urban, rural, etc.)
            object_inventory: Dict of {object_name: count}
            layout_description: Description of spatial arrangement
            appearance_details: Visual characteristics
            lighting_conditions: Lighting description
            spatial_relationships: List of spatial relationships
        
        Returns:
            Optimized SD prompt string
        """
        # Expand scene type
        scene = self.SCENE_EXPANSIONS.get(scene_type.lower(), scene_type)
        
        # Build layout with spatial relationships
        layout = layout_description
        if spatial_relationships:
            layout += ". " + ", ".join(spatial_relationships[:5])
        
        # Build objects string with counts and positions
        objects_parts = []
        for obj, count in object_inventory.items():
            count_word = self.COUNT_WORDS.get(min(count, 10), "multiple")
            if count > 1:
                objects_parts.append(f"{count_word} {obj}s")
            else:
                objects_parts.append(f"{count_word} {obj}")
        objects = ", ".join(objects_parts) if objects_parts else "various structures"
        
        # Format prompt
        prompt = self.SD_PROMPT_TEMPLATE.format(
            scene=scene,
            layout=layout,
            objects=objects,
            appearance=appearance_details or "natural satellite imagery colors",
            lighting=lighting_conditions or "daylight conditions"
        )
        
        return prompt
    
    def generate_refined_caption(
        self,
        scene_type: str,
        object_inventory: Dict[str, int],
        spatial_relationships: Optional[List[str]] = None,
        appearance_details: Optional[str] = None,
        detection_labels: Optional[List[str]] = None
    ) -> str:
        """
        Generate a refined caption for training.
        
        Args:
            scene_type: Type of scene
            object_inventory: Dict of {object_name: count}
            spatial_relationships: List of spatial relationships
            appearance_details: Visual details
            detection_labels: Labels from detection
        
        Returns:
            Refined caption string
        """
        # Scene description
        scene_expanded = self.SCENE_EXPANSIONS.get(scene_type.lower(), scene_type)
        scene_desc = f"Aerial view of {scene_expanded} area"
        
        # Objects with counts
        if object_inventory:
            obj_parts = []
            for obj, count in list(object_inventory.items())[:5]:
                if count > 1:
                    obj_parts.append(f"{count} {obj}s")
                else:
                    obj_parts.append(obj)
            objects = ", ".join(obj_parts)
        elif detection_labels:
            objects = ", ".join(set(detection_labels[:5]))
        else:
            objects = "various structures"
        
        # Spatial context
        if spatial_relationships:
            spatial_context = spatial_relationships[0]
        else:
            spatial_context = "Objects distributed across the scene"
        
        # Visual details (shortened)
        if appearance_details:
            visual_details = appearance_details[:100]
        else:
            visual_details = "Natural colors and clear visibility"
        
        # Build caption
        caption = self.CAPTION_TEMPLATE.format(
            scene_desc=scene_desc,
            objects=objects,
            spatial_context=spatial_context,
            visual_details=visual_details
        )
        
        # Clean up
        caption = re.sub(r'\s+', ' ', caption).strip()
        caption = caption.replace('..', '.')
        
        return caption
    
    def refine_crop_caption(
        self,
        object_label: str,
        position: str,
        surrounding_objects: List[str],
        spatial_context: str,
        qwen_description: str,
        relative_size: str = "medium"
    ) -> str:
        """
        Generate refined caption for a zoom crop.
        
        Args:
            object_label: Primary object label
            position: Position in image (e.g., "top-left")
            surrounding_objects: List of nearby objects
            spatial_context: Spatial relationship description
            qwen_description: Qwen2-VL generated description
            relative_size: Size relative to image
        
        Returns:
            Refined caption for the crop
        """
        parts = []
        
        # Object with size modifier
        if relative_size == "large":
            parts.append(f"Large {object_label}")
        elif relative_size == "small":
            parts.append(f"Small {object_label}")
        else:
            parts.append(object_label.capitalize())
        
        # Extract descriptors from Qwen caption
        descriptors = self._extract_descriptors(qwen_description)
        if descriptors:
            parts[0] = f"{', '.join(descriptors[:2])} {parts[0].lower()}"
        
        # Position
        position_formatted = position.replace("-", " ").replace("middle", "center")
        parts.append(f"in {position_formatted} of aerial scene")
        
        # Surrounding context
        if surrounding_objects:
            if len(surrounding_objects) == 1:
                parts.append(f"adjacent {surrounding_objects[0]} nearby")
            else:
                parts.append(f"near {' and '.join(surrounding_objects[:2])}")
        
        # Spatial relationships
        if spatial_context and spatial_context != "isolated":
            parts.append(spatial_context)
        
        caption = ", ".join(parts) + "."
        caption = caption[0].upper() + caption[1:]
        
        return caption
    
    def _extract_descriptors(self, text: str) -> List[str]:
        """Extract descriptive words from text."""
        if not text:
            return []
        
        descriptors = []
        text_lower = text.lower()
        
        # Shape descriptors
        shapes = ["rectangular", "square", "circular", "round", "triangular", "irregular"]
        for shape in shapes:
            if shape in text_lower:
                descriptors.append(shape)
        
        # Material/surface descriptors
        materials = ["concrete", "metal", "metallic", "wooden", "brick", "glass", "asphalt"]
        for material in materials:
            if material in text_lower:
                descriptors.append(material)
        
        # Color descriptors
        colors = ["white", "gray", "grey", "brown", "green", "blue", "red", "dark", "light"]
        for color in colors:
            if color in text_lower:
                descriptors.append(color)
        
        # Roof descriptors (for buildings)
        roofs = ["flat roof", "sloped roof", "pitched roof", "dome"]
        for roof in roofs:
            if roof in text_lower:
                descriptors.append(roof.replace(" ", "-"))
        
        return descriptors[:3]  # Limit to 3 descriptors
    
    def compare_extractions(
        self,
        original_extraction: Dict[str, Any],
        verification_extraction: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare original and verification extractions to assess quality.
        
        Args:
            original_extraction: First-pass extraction results
            verification_extraction: Second-pass verification results
        
        Returns:
            Dict of comparison metrics
        """
        metrics = {}
        
        # Scene type consistency
        orig_scene = original_extraction.get("scene_type", "").lower()
        verify_scene = verification_extraction.get("scene_type", "").lower()
        
        if orig_scene == verify_scene:
            metrics["scene_consistency"] = 1.0
        elif orig_scene in verify_scene or verify_scene in orig_scene:
            metrics["scene_consistency"] = 0.7
        else:
            metrics["scene_consistency"] = 0.3
        
        # Object inventory overlap
        orig_objects = set(original_extraction.get("object_inventory", {}).keys())
        verify_objects = set(verification_extraction.get("object_inventory", {}).keys())
        
        if orig_objects and verify_objects:
            intersection = orig_objects & verify_objects
            union = orig_objects | verify_objects
            metrics["object_overlap"] = len(intersection) / len(union) if union else 0.5
        else:
            metrics["object_overlap"] = 0.5
        
        # Count accuracy
        orig_counts = original_extraction.get("object_inventory", {})
        verify_counts = verification_extraction.get("object_inventory", {})
        
        if orig_counts:
            total_orig = sum(orig_counts.values())
            total_verify = sum(verify_counts.values())
            
            if total_orig > 0 and total_verify > 0:
                ratio = min(total_orig, total_verify) / max(total_orig, total_verify)
                metrics["count_accuracy"] = ratio
            else:
                metrics["count_accuracy"] = 0.5
        else:
            metrics["count_accuracy"] = 0.5
        
        # Overall consistency score
        metrics["overall_consistency"] = (
            metrics["scene_consistency"] * 0.3 +
            metrics["object_overlap"] * 0.4 +
            metrics["count_accuracy"] * 0.3
        )
        
        return metrics
    
    def generate_verification_prompt(self, generated_image_analysis: Dict[str, Any]) -> str:
        """
        Generate a final caption based on verification analysis.
        
        This ensures the caption perfectly describes the ACTUAL generated image,
        not what we intended to generate.
        
        Args:
            generated_image_analysis: Analysis results from second-pass Qwen2-VL
        
        Returns:
            Verified caption that matches the generated image
        """
        scene_type = generated_image_analysis.get("scene_type", "aerial scene")
        objects = generated_image_analysis.get("object_inventory", {})
        layout = generated_image_analysis.get("layout_description", "")
        appearance = generated_image_analysis.get("appearance_details", "")
        
        # Generate caption from what we ACTUALLY see in the generated image
        caption = self.generate_refined_caption(
            scene_type=scene_type,
            object_inventory=objects,
            spatial_relationships=[layout] if layout else None,
            appearance_details=appearance
        )
        
        return caption


# Convenience functions
def create_sd_prompt(
    scene_type: str,
    objects: Dict[str, int],
    layout: str = "",
    appearance: str = "",
    lighting: str = ""
) -> str:
    """Quick helper to create SD prompt."""
    engine = PromptRefinementEngine()
    return engine.generate_sd_prompt(
        scene_type=scene_type,
        object_inventory=objects,
        layout_description=layout,
        appearance_details=appearance,
        lighting_conditions=lighting
    )


def create_refined_caption(
    scene_type: str,
    objects: Dict[str, int],
    spatial: Optional[List[str]] = None
) -> str:
    """Quick helper to create refined caption."""
    engine = PromptRefinementEngine()
    return engine.generate_refined_caption(
        scene_type=scene_type,
        object_inventory=objects,
        spatial_relationships=spatial
    )
