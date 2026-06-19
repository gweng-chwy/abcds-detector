# Shorts Feature Definitions

Source: `features_repository/shorts_features.py`.

Each feature below includes every `VideoFeature` config attribute currently defined in code.

## Attract

### `tight_framing_index` - Tight Framing & Visual Dominance

| Attribute | Value |
|---|---|
| `id` | `tight_framing_index` |
| `name` | Tight Framing & Visual Dominance |
| `category` | `SHORTS` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies the spatial dominance of the primary subject.<br>Tight framing is defined by a Subject-to-Frame Ratio (SfR) of ≥60%.<br>The score reflects the 'Density' (persistence) of tight framing, <br>differentiating between incidental close-ups and thematic visual dominance.</code></pre> |
| `prompt_template` | <pre><code>Act as a professional Cinematographer and Video Analyst. Your goal is to measure<br>'Visual Weight' through Tight Framing detection.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. QUANTITATIVE HEURISTICS:<br>- **Extreme Close-Up (ECU):** Subject fills &gt;80% of frame.<br>- **Close-Up (CU):** Subject fills 60% - 80% of frame.<br>- **Medium Shot (MS):** Subject fills 30% - 59% of frame.<br>- **Wide/Long Shot (LS):** Subject fills &lt;30% of frame.<br><br>### 2. DENSITY &amp; QUALITY LOGIC:<br>- **Density Score:** (Total duration of all CU and ECU shots) / (Total video duration).<br>Represented as density_score in the JSON.<br>- **Feature Quality Score:** Map the density_score to the following scale:<br>    * 0.9-1.0: Density &gt; 70% (Dominant)<br>    * 0.7-0.8: Density 40-70% (Strong)<br>    * 0.4-0.6: Density 20-40% (Balanced)<br>    * 0.1-0.3: Density &lt; 20% (Incidental)<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float, # Certainty of visual detection accuracy<br>    "feature_quality_score": float, # The 0.0-1.0 score based on the Density Scale<br>    "metrics": {{<br>        "density_score": float, # MANDATORY: Total tight-frame duration / Total duration<br>        "peak_sfr_percentage": float, # Highest Subject-to-Frame ratio observed<br>        "primary_subject_class": str, # "Product", "Human_Face", "Text", "Abstract"<br>        "framing_cadence": "Static" &#124; "Fast-Cutting" &#124; "Zoom-In-Progressive"<br>    }},<br>    "spatial_analysis": {{<br>        "average_negative_space_ratio": float, # 1.0 - average SfR<br>        "edge_collision": boolean, # Subject bleeds off the edges<br>        "occlusion_level": "None" &#124; "Partial" &#124; "Heavy"<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "shot_type": "ECU" &#124; "CU",<br>            "subject_dominance_score": float,<br>            "description": str<br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "visual_impact_score": float, # Effectiveness for mobile viewing<br>        "is_hook_tightly_framed": boolean, # Evaluates the first 3 seconds<br>        "summary": "Concise technical summary of framing strategy"<br>    }}<br>}}<br><br>### EVALUATION LOGIC:<br>- **The Hook:** If the first 3 seconds are tight-framed, increase the visual_impact_score.<br>- **Negative Space:** Ignore solid color backgrounds (graphics); focus on the subject's physical bounding box.<br>- **Calculation:** Ensure the density_score is a precise float based on the temporal_segments sum.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_human_voice` - Human Voice Presence

| Attribute | Value |
|---|---|
| `id` | `shorts_human_voice` |
| `name` | Human Voice Presence |
| `category` | `SHORTS` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies the presence, duration, and quality of human speech. <br>Voice includes Voice-Overs (VO), direct-to-camera dialogue, or background <br>narration. The metric measures 'Vocal Density' (percentage of video containing <br>speech) and assesses the clarity and role of the speaker.</code></pre> |
| `prompt_template` | <pre><code>Act as a professional Cinematographer and Video Analyst. Your goal is to analyze the audio track of this video specifically for human vocal presence.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Density Score:** (Total duration of audible human speech) / (Total video duration). It is referred to as density_score in the JSON file.<br>*Example: If there is a 2s intro greeting and a 3s closing call-to-action in a 10s video, Density Score is 0.5.*<br>- **Vocal Clarity:** The ease with which the voice is understood (0.0 - 1.0). High score = studio quality/clear; Low score = muffled, heavy background noise, or distorted.<br><br>### 2. VOCAL CATEGORIES:<br>- **Voice Over (VO):** Narrative voice added in post-production.<br>- **Dialogue:** On-camera person speaking.<br>- **Ambient Speech:** Overheard background talking.<br>- **Synthetic/AI Voice:** Clear AI-generated narration (text-to-speech).<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean, <br>    "confidence_score": float, # Certainty that the detected audio is a human voice<br>    "metrics": {{<br>        "density_score": float, # Total speech duration / Total video duration<br>        "vocal_clarity_score": float, # 0.0 to 1.0 based on signal-to-noise ratio<br>        "primary_voice_type": "Voice_Over" &#124; "Dialogue" &#124; "AI_Synthetic" &#124; "Mixed",<br>        "speech_cadence": "Constant" &#124; "Intermittent" &#124; "Rapid" &#124; "Slow"<br>    }},<br>    "audio_analysis": {{<br>        "background_noise_level": "Low" &#124; "Medium" &#124; "High",<br>        "music_overlap_interference": boolean, # Does background music drown out the voice?<br>        "speaker_gender_estimate": "Male" &#124; "Female" &#124; "Multiple" &#124; "N/A"<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "voice_role": "Narration" &#124; "Hook" &#124; "CTA" &#124; "Ambient",<br>            "clarity_rating": float,<br>            "description": str # e.g., "Creator introduces product features"<br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "vocal_impact_score": float, # How much does the voice drive the narrative?<br>        "is_hook_voiced": boolean, # Does speech start within the first 1.5 seconds?<br>        "summary": "Technical summary of audio/vocal strategy"<br>    }}<br>}}<br><br>### EVALUATION STEPS:<br>1. Identify all time stamps where a human (or AI) voice is audible.<br>2. Calculate the **density_score** (Total Speech Time / Total Duration).<br>3. Evaluate the **vocal_clarity_score**—reduce this if background music or noise makes speech hard to follow.<br>4. Determine if the "Hook" (0:00-0:02) contains speech, as this is a high-retention signal.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_direct_to_camera` - Direct to Camera

| Attribute | Value |
|---|---|
| `id` | `shorts_direct_to_camera` |
| `name` | Direct to Camera |
| `category` | `SHORTS` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies the duration and intensity of direct eye contact between the <br>on-screen subject and the camera lens. This feature measures the <br>'Direct Address Density' and assesses the intimacy of the framing <br>(e.g., face-to-face address).</code></pre> |
| `prompt_template` | <pre><code>Act as a professional Cinematographer and Video Analyst. Your goal is to <br>analyze the video for instances where a person looks directly into the camera lens <br>to address the viewer.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Density Score:** (Total duration of direct eye contact / address) / (Total video duration). <br>Represented as feature_quality_score in the JSON.<br>- **Eye Contact Intensity:** A measure of how consistently the subject maintains <br>gaze without looking away at scripts or monitors (0.0 - 1.0).<br><br>### 2. ADDRESS MODES:<br>- **Direct Address:** Subject is looking into the lens and speaking.<br>- **Silent Gaze:** Subject maintains eye contact without speaking (e.g., reacting to a sound).<br>- **Glance:** Brief, intermittent eye contact (less than 0.5s).<br>- **Off-Camera:** Subject is looking at a secondary point (3/4 profile), not the viewer.<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean, <br>    "confidence_score": float, # Certainty that gaze is directed at the lens<br>    "feature_quality_score": float, # Total direct address duration / Total duration<br>    "metrics": {{<br>        "eye_contact_intensity": float, # 0.0 to 1.0 (steadiness of gaze)<br>        "subject_distance": "Close-Up" &#124; "Medium" &#124; "Full-Body",<br>        "address_style": "Personal/Intimate" &#124; "Presentational" &#124; "Accidental"<br>    }},<br>    "visual_engagement_analysis": {{<br>        "facial_visibility": "Full" &#124; "Partial" &#124; "Occluded",<br>        "eye_level_alignment": "Eye-Level" &#124; "High-Angle" &#124; "Low-Angle",<br>        "emotional_delivery": str # e.g., "High-energy", "Authentic", "Stoic"<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "gaze_type": "Direct_Address" &#124; "Silent_Gaze" &#124; "Intermittent",<br>            "eye_contact_strength": float,<br>            "description": str # e.g., "Speaker addresses viewer during the hook"<br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "parasocial_score": float, # How effectively does the subject "connect" with the viewer?<br>        "is_hook_direct": boolean, # Does direct eye contact occur in the first 1.5 seconds?<br>        "summary": "Technical summary of direct address strategy"<br>    }}<br>}}<br><br>### EVALUATION STEPS:<br>1. Identify all segments where the subject's pupils are directed at the camera lens.<br>2. Calculate the **feature_quality_score** by dividing direct address time by total duration.<br>3. Assess **eye_contact_intensity**—lower this if the subject is clearly reading a teleprompter or looking at themselves on the phone screen rather than the lens.<br>4. Check the "Hook" (0:00-0:02); direct address at the start is a major retention driver.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_has_supers` - Supers & Text-Audio Synchronicity

| Attribute | Value |
|---|---|
| `id` | `shorts_has_supers` |
| `name` | Supers & Text-Audio Synchronicity |
| `category` | `SHORTS` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies the presence, accuracy, and synchronization of text overlays (supers) <br>with the spoken audio. This measures 'Text Density' and the 'Synchronicity Score' <br>to determine how effectively the visual text reinforces the spoken message.</code></pre> |
| `prompt_template` | <pre><code>Act as a professional Cinematographer and Video Analyst. Your goal is to <br>Analyze the video for SUPERS (text overlays) and their relationship to the audio.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Density Score:** (Total duration where text overlays are visible) / (Total video duration).<br>  Represented as feature_quality_score in the JSON.<br>- **Synchronicity Score:** (0.0 - 1.0) measure of how well text timing matches spoken words. <br>  1.0 = frame-perfect captions; 0.5 = static text roughly related; 0.0 = no relation.<br><br>### 2. SUPERS CATEGORIES:<br>- **Dynamic Captions:** Word-by-word or phrase-by-phrase synced text.<br>- **Static Callouts:** Persistent text (e.g., "50% OFF" or "Product Name").<br>- **Kinetic Typography:** Stylized, moving text used for emphasis.<br>- **Headlines:** Large top/bottom text bars that stay throughout the video.<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float, # Certainty of text detection<br>    "feature_quality_score": float, # 0.0-1.0 Time text is visible / Total duration<br>    "metrics": {{<br>        "synchronicity_score": float, # Match between audio and text timing<br>        "text_coverage_ratio": float, # Percentage of frame area occupied by text<br>        "primary_supers_type": "Dynamic_Captions" &#124; "Static_Callouts" &#124; "Headlines" &#124; "Mixed"<br>    }},<br>    "visual_analysis": {{<br>        "readability_score": float, # Contrast and font clarity (0.0 - 1.0)<br>        "is_mobile_safe": boolean, # Is text clear of UI elements (likes/captions)?<br>        "font_style": "Minimal" &#124; "Bold/Aggressive" &#124; "Stylized/Brand"<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "text_content": str,<br>            "matches_audio": boolean,<br>            "style": "Caption" &#124; "Emphasis" &#124; "CTA"<br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "narrative_reinforcement_score": float, # How well text aids understanding<br>        "is_hook_text_present": boolean, # Does text appear in the first 1.5 seconds?<br>        "summary": "Technical summary of text overlay strategy"<br>    }}<br>}}<br><br>### EVALUATION STEPS:<br>1. Identify all segments where text is overlaid on the video.<br>2. Compare the text content against the audio track for verbatim or supportive matching.<br>3. Calculate the **feature_quality_score** based on text visibility duration.<br>4. Assess the **synchronicity_score**—lower this if text lingers too long or appears after the audio has passed.<br>5. Verify if text is in the "Mobile Safe Zone" (central area, not blocked by platform UI).</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

## Brand

### `shorts_product_closeup` - Product Close-Up

| Attribute | Value |
|---|---|
| `id` | `shorts_product_closeup` |
| `name` | Product Close-Up |
| `category` | `SHORTS` |
| `sub_category` | `BRAND` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies segments where the product occupies at least 30% of the frame. <br>This measures standard product visibility and presence within a recognizable <br>context or environment.</code></pre> |
| `prompt_template` | <pre><code>Act as a professional Cinematographer and Video Analyst. Your goal is to <br>measure 'Product Presence' via Close-Up detection.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Product Close-Up (CU):** Product occupies 30% to 59% of the frame area.<br>- **Density Score:** (Total duration of Product CU shots) / (Total video duration).<br>Represented as feature_quality_score in the JSON.<br><br>### 2. DYNAMIC SCORING (0.0 - 1.0):<br>- 0.9-1.0: Product CU is the primary visual anchor (Density &gt; 60%).<br>- 0.6-0.8: Product is featured in CU at key intervals (Density 30-60%).<br>- 0.1-0.5: Product CU is incidental or brief (Density &lt; 30%).<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float, <br>    "feature_quality_score": float, # Based on the Dynamic Scoring scale<br>    "metrics": {{<br>        "density_score": float, # MANDATORY: CU duration / Total duration<br>        "average_sfr_percentage": float, # Average Subject-to-Frame ratio for CU shots<br>        "product_identifiability": float, # Clarity of branding (0.0 - 1.0)<br>        "framing_style": "Handheld" &#124; "Studio-Static" &#124; "Pan/Tilt"<br>    }},<br>    "spatial_analysis": {{<br>        "rule_of_thirds_align": boolean,<br>        "background_distraction_level": "Low" &#124; "Medium" &#124; "High",<br>        "is_product_centered": boolean<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "sfr_percentage": float,<br>            "description": str <br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "visual_impact_score": float,<br>        "is_hook_product_featured": boolean, # Product CU in first 2 seconds?<br>        "summary": "Technical summary of product CU strategy"<br>    }}<br>}}<br><br>### EVALUATION LOGIC:<br>- Only count segments where the Product is the main subject and fills 30%-59% of the frame.<br>- If the product fills &gt;60%, it exceeds this feature and belongs in the 'Extreme' category.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_product_extreme_closeup` - Product Extreme Close-Up

| Attribute | Value |
|---|---|
| `id` | `shorts_product_extreme_closeup` |
| `name` | Product Extreme Close-Up |
| `category` | `SHORTS` |
| `sub_category` | `BRAND` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies segments where the product is the dominant visual element, <br>occupying 60% or more of the frame. This measures 'Macro' focus and <br>high-detail product showcasing.</code></pre> |
| `prompt_template` | <pre><code>Act as a professional Cinematographer and Video Analyst. Your goal is to <br>measure 'Product Dominance' via Extreme Close-Up (ECU) detection.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Product Extreme Close-Up (ECU):** Product occupies 60% or more of the frame area.<br>- **Density Score:** (Total duration of Product ECU shots) / (Total video duration).<br>Represented as feature_quality_score in the JSON.<br><br>### 2. DYNAMIC SCORING (0.0 - 1.0):<br>- 0.9-1.0: Macro-heavy edit; high detail focus (Density &gt; 40%).<br>- 0.6-0.8: Strategic use of macro shots for texture/detail (Density 15-40%).<br>- 0.1-0.5: Incidental or very brief macro moments (Density &lt; 15%).<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "feature_quality_score": float, <br>    "metrics": {{<br>        "density_score": float, # MANDATORY: ECU duration / Total duration<br>        "peak_sfr_percentage": float, # Max frame fill observed<br>        "texture_visibility": "Low" &#124; "Medium" &#124; "High", # Does it show material detail?<br>        "lighting_quality": "Flat" &#124; "Cinematic" &#124; "Overexposed"<br>    }},<br>    "spatial_analysis": {{<br>        "edge_collision": boolean, # Does the product extend beyond frame edges?<br>        "depth_of_field": "Shallow" &#124; "Deep", # Is the background blurred?<br>        "focal_point": str # e.g., "Logo", "Texture", "Nozzle", "Screen"<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "sfr_percentage": float,<br>            "focus_point": str,<br>            "description": str <br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "visual_impact_score": float,<br>        "is_macro_hook": boolean, # Does the video open with a macro shot?<br>        "summary": "Technical summary of product ECU/Macro strategy"<br>    }}<br>}}<br><br>### EVALUATION LOGIC:<br>- Only count segments where the Product fills 60% or more of the frame.<br>- Focus on detail: ECU shots are intended to show the "hero" aspects of the product.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `e_shorts_brand_mention_speech_1st_5_secs` - Brand Mention (Speech) (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `e_shorts_brand_mention_speech_1st_5_secs` |
| `name` | Brand Mention (Speech) (First 5 seconds) |
| `category` | `SHORTS` |
| `sub_category` | `BRAND` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>The brand name is heard in the audio or speech in the first 5 seconds of the video.</code></pre> |
| `prompt_template` | <pre><code>Does the speech mention the brand {brand_name} in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Only evaluate audio or speech from the first 5 seconds of the video.<br><br>3. Count clear mentions of the brand name or any configured brand variation.<br><br>4. Provide the exact timestamp when the brand {brand_name} is heard in speech.<br><br>5. Return True if and only if the brand {brand_name} or a configured brand variation is heard in the first 5 seconds. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

## Connect

### `shorts_product_context_index` - Product Context & Usage Quality

| Attribute | Value |
|---|---|
| `id` | `shorts_product_context_index` |
| `name` | Product Context & Usage Quality |
| `category` | `SHORTS` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Evaluates the 'Show, Don't Tell' quality. Quantifies physical <br>interaction, environmental realism, and utility demonstration. <br>Measures both the duration of usage (Density) and the effectiveness <br>of the demonstration (Quality Score).   </code></pre> |
| `prompt_template` | <pre><code>Act as a professional Cinematographer and Video Analyst. <br>Analyze the 'Product-in-Use' effectiveness.<br><br>VIDEO METADATA:<br>{metadata_summary}<br><br>SCORE ON THREE DIMENSIONS (0-100 each):<br><br>1. INTERACTION DEPTH (40%)<br>- Focus: Physical contact and active engagement.<br>- Criteria: Is the product being handled, worn, consumed, or operated?<br>- 90-100: Detailed, multi-step interaction or sustained use (&gt;4s).<br>- 70-89: Clear physical interaction but brief (2-4s).<br>- 0-69: Minimal touching or product is merely a prop in the frame.<br><br>2. CONTEXTUAL REALISM (30%)<br>- Focus: The "Where" and "Who". <br>- Criteria: Is the environment a "lived-in" space (home, gym, office) vs. a sterile studio? <br>- 90-100: Authentic daily-life setting with natural lighting and relatable user behavior.<br>- 70-89: Recognizable setting but feels slightly "staged" or over-polished.<br>- 0-69: Infomercial style, white backgrounds, or disconnected from reality.<br><br>3. UTILITY DEMONSTRATION (30%)<br>- Focus: The "Why".<br>- Criteria: Does the interaction show the product's purpose/benefit without needing a voiceover?<br>- 90-100: The usage clearly solves a pain point or achieves a goal (e.g., thirst quenched, app task finished).<br>- 70-89: Product is used correctly but the "benefit" is implied rather than obvious.<br>- 0-69: Random interaction that doesn't showcase what the product actually does.<br><br>FINAL CALCULATION:<br>Overall Score = (Interaction * 0.4) + (Realism * 0.3) + (Utility * 0.3)<br><br>FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "evaluation": {{<br>        "interaction_depth": {{<br>            "score": int,<br>            "action_type": "physical&#124;consumption&#124;digital&#124;service",<br>            "duration": float,<br>            "evidence": str<br>        }},<br>        "contextual_realism": {{<br>            "score": int,<br>            "environment": str,<br>            "authenticity_level": "natural&#124;staged&#124;studio",<br>            "observation": str<br>        }},<br>        "utility_demo": {{<br>            "score": int,<br>            "benefit_shown": str,<br>            "clarity": "explicit&#124;implicit&#124;none"<br>        }},<br>        "final_scoring": {{<br>            "interaction_weighted": float,<br>            "realism_weighted": float,<br>            "utility_weighted": float,<br>            "total_score": float # Sum of the three weighted scores above<br>        }}<br>    }}<br>}}<br><br>SCORING GUIDANCE:<br>- HIGH (80+): A "Day-in-the-life" feel. User solves a problem using the product in a real room.<br>- MED (50-79): Product is used, but it feels like a "commercial." Correct use, but overly scripted.<br>- LOW (&lt;50): Product is just sitting there, or being held like a trophy for the camera.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_casual_language` - Casual Language

| Attribute | Value |
|---|---|
| `id` | `shorts_casual_language` |
| `name` | Casual Language |
| `category` | `SHORTS` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies the informality of the script. Measures the use of everyday language, <br>slang, contractions, and conversational filler vs. formal/corporate scripted speech.</code></pre> |
| `prompt_template` | <pre><code>Act as a Linguistic and  Video Analyst. Your goal is to measure 'Tone Informality.'<br><br>VIDEO METADATA:<br>{metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Density Score:** (Duration of conversational/casual speech) / (Total speech duration).<br>- **Informality Rating:** (0.0 - 1.0) 1.0 = "POV/FaceTime" style; 0.5 = Standard commercial; 0.0 = Corporate/Medical.<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "feature_quality_score": float,<br>    "metrics": {{<br>        "density_score": float,<br>        "slang_presence": boolean,<br>        "filler_word_frequency": "Low" &#124; "Medium" &#124; "High", # e.g., "um", "like", "literally"<br>        "script_type": "Ad-lib/Spontaneous" &#124; "Conversational-Scripted" &#124; "Formal"<br>    }},<br>    "overall_assessment": {{<br>        "authenticity_score": float, # How 'real' does the speech feel?<br>        "summary": "Analysis of linguistic tone"<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_humor_index` - Humor & Comedic Timing

| Attribute | Value |
|---|---|
| `id` | `shorts_humor_index` |
| `name` | Humor & Comedic Timing |
| `category` | `SHORTS` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Detects and quantifies attempts at humor, including wit, physical comedy, <br>satire, or comedic timing.</code></pre> |
| `prompt_template` | <pre><code>Act as a Creative Strategist. Analyze the video for 'Comedic Intent.'<br><br>    VIDEO METADATA:<br>    {metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Density Score:** (Duration of comedic setups/payoffs) / (Total video duration).<br>- **Humor Type:** "Observational", "Slapstick", "Deadpan", "Satirical".<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "feature_quality_score": float,<br>    "metrics": {{<br>        "density_score": float,<br>        "humor_mechanism": str, # e.g., "Visual gag", "Funny VO", "Reaction"<br>        "edge_factor": float # 0.0 (Safe) to 1.0 (Risky/Bold)<br>    }},<br>    "overall_assessment": {{<br>        "entertainment_value": float,<br>        "is_hook_funny": boolean,<br>        "summary": "Technical summary of humor strategy"<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `character_driven` - Character-Driven

| Attribute | Value |
|---|---|
| `id` | `character_driven` |
| `name` | Character-Driven |
| `category` | `SHORTS` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Video features a relatable character whose journey or transformation resonates with audience.<br>Evaluates character prominence, relatability, and narrative journey shown.</code></pre> |
| `prompt_template` | <pre><code>Act as a Narrative Strategist and Video Analyst. Your goal is to measure <br>'Character Dominance' and 'Persona-Led Storytelling.'<br><br>VIDEO METADATA:<br>{metadata_summary}<br><br>SCORE ON THREE DIMENSIONS (0-100 each):<br><br>1. CHARACTER PROMINENCE (40% weight)<br>   Character is clear protagonist with distinct personality/role<br>   Look for: Clear lead character, distinct personality traits, on-screen presence<br>   - 90-100: Strong, well-developed protagonist<br>   - 70-89: Clear character with distinct personality<br>   - 50-69: Character present but underdeveloped<br>   - 0-49: No clear character focus<br><br>2. CHARACTER JOURNEY/TRANSFORMATION (30% weight)<br>   Character shows visible journey, change, or problem-solving<br>   Look for: Before/after transformation, challenge faced, goal achieved, emotional arc<br>   - 90-100: Clear narrative journey with visible transformation<br>   - 70-89: Character shows clear journey or growth<br>   - 50-69: Some journey element but subtle<br>   - 0-49: No journey or transformation shown<br><br>3. AUDIENCE RELATABILITY (30% weight)<br>   Character is relatable to target audience (emotional, realistic, authentic)<br>   Look for: Authentic emotion, realistic situation, audience alignment, genuine engagement<br>   - 90-100: Highly relatable, authentic emotion<br>   - 70-89: Mostly relatable character<br>   - 50-69: Somewhat relatable<br>   - 0-49: Not relatable or inauthentic<br><br>FINAL CALCULATION:<br>Overall Score = (Prominence × 0.40) + (Journey × 0.30) + (Relatability × 0.30)<br><br>FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "evaluation": {{<br>        "character_present": boolean,<br>        "character_type": str,<br>        "personality_traits": [str],<br>        "journey_type": str,<br>        "prominence_score": int,<br>        "journey_score": int,<br>        "relatability_score": int,<br>        "weighted_overall": float<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

## Direct

### `shorts_audio_cta` - Call to Action (Audio)

| Attribute | Value |
|---|---|
| `id` | `shorts_audio_cta` |
| `name` | Call to Action (Audio) |
| `category` | `SHORTS` |
| `sub_category` | `DIRECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Detects and quantifies spoken instructions that direct the viewer to take <br>action. This includes verbal commands from Voice-Overs (VO) or on-screen talent. <br>Measures 'CTA Density' and 'Urgency Level' to determine the strength of the <br>conversion signal.</code></pre> |
| `prompt_template` | <pre><code>Act as a Direct Response Marketing Analyst. Your goal is to identify and <br>quantify the spoken Call to Action (CTA).<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Density Score:** (Total duration of the spoken CTA) / (Total video duration).<br>  Represented as density_score in the JSON.<br>- **CTA Urgency:** (0.0 - 1.0) 1.0 = Explicit command with time-sensitivity (e.g., "Click the link below."); <br>  0.5 = General suggestion (e.g., "Check us out"); 0.1 = Brand mention only.<br><br>### 2. CTA DELIVERY MODES:<br>- **Direct Address:** On-screen talent looks at camera and gives the CTA.<br>- **Voice-Over (VO):** Narrator delivers the CTA over b-roll or product shots.<br>- **Off-Camera:** Secondary character or background voice mentions the action.<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float, # Certainty that a verbal CTA was issued<br>    "feature_quality_score": float, # 0.0-1.0 based on clarity and persuasiveness<br>    "metrics": {{<br>        "density_score": float, # MANDATORY: CTA duration / Total video duration<br>        "cta_urgency_score": float, # 0.0 to 1.0<br>        "delivery_method": "On-Screen_Talent" &#124; "Voice-Over" &#124; "Mixed",<br>        "cta_type": "Hard_Sell" &#124; "Soft_Suggestion" &#124; "Inspirational"<br>    }},<br>    "linguistic_analysis": {{<br>        "verbatim_text": str, # The exact words used for the CTA<br>        "placement_type": "End-Roll" &#124; "Mid-Roll" &#124; "Early-Hook",<br>        "contains_incentive": boolean # e.g., "Use code SAVE10", "Free shipping"<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "cta_content": str,<br>            "loudness_relative_to_avg": "Quieter" &#124; "Normal" &#124; "Emphasized"<br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "conversion_potential": float, # How likely is this audio to drive a click?<br>        "is_cta_at_end": boolean, # Does the audio end on a CTA?<br>        "summary": "Technical analysis of verbal CTA effectiveness"<br>    }}<br>}}<br><br>### EVALUATION LOGIC:<br>1. Scan the audio track for imperative verbs (Shop, Buy, Click, Visit, Try, Download).<br>2. Calculate the **density_score** by summing the duration of these verbal triggers.<br>3. Assess **feature_quality_score** based on vocal clarity and "The Ask"—if the CTA is muffled or buried under loud music, lower the score.<br>4. Note the placement: A CTA at the very end is standard; a CTA in the first 5 seconds is a "Fast-Action" strategy.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `special_offer_speech` - Special Offer (Speech)

| Attribute | Value |
|---|---|
| `id` | `special_offer_speech` |
| `name` | Special Offer (Speech) |
| `category` | `SHORTS` |
| `sub_category` | `DIRECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Audio/voiceover explicitly announces special offer, discount, or deal.<br>Evaluates clarity of offer type, specific details mentioned, and delivery emphasis.</code></pre> |
| `prompt_template` | <pre><code>Act as a Direct Response Marketing Analyst. Your goal is to evaluate: Is there a SPECIAL OFFER announced in speech?<br><br>VIDEO METADATA:<br>{metadata_summary}<br><br>SCORE ON THREE DIMENSIONS (0-100 each):<br><br>1. OFFER TYPE CLARITY (40% weight)<br>   Clear announcement of what offer is (discount %, deal, promotion type)<br>   Look for: "20% off", "Free with purchase", "Buy one get one", "Limited time deal"<br>   - 90-100: Very clear offer type with specific details (e.g., "20% off")<br>   - 70-89: Clear offer mentioned with reasonable detail<br>   - 50-69: Offer mentioned but details vague or implied<br>   - 0-49: No offer details in audio<br><br>2. DELIVERY EMPHASIS (35% weight)<br>   Strong emphasis given through voice tone, repetition, or prominence<br>   Look for: Emphatic tone, repeated mention, early in spot, vocal excitement<br>   - 90-100: Strong emphasis with enthusiastic delivery<br>   - 70-89: Clear emphasis given in delivery<br>   - 50-69: Mentioned with moderate emphasis<br>   - 0-49: Buried or casual mention<br><br>3. OFFER PROMINENCE (25% weight)<br>   Offer featured continuously or at strategic moments in video<br>   Look for: Multiple mentions, mentioned early, highlighted throughout<br>   - 90-100: Prominent throughout, multiple emphatic mentions<br>   - 70-89: Clear prominence in video<br>   - 50-69: Mentioned but not prominent<br>   - 0-49: Single casual mention<br><br>FINAL CALCULATION:<br>Overall Score = (OfferClarity × 0.40) + (Emphasis × 0.35) + (Prominence × 0.25)<br><br>FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "evaluation": {{<br>        "offer_type": str,<br>        "offer_details": str,<br>        "delivery_tone": str,<br>        "mention_count": int,<br>        "offer_clarity_score": int,<br>        "emphasis_score": int,<br>        "prominence_score": int,<br>        "weighted_overall": float<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

## Other / None

### `e_shorts_max_10_words_per_frame` - Max 10 Words Per Frame

| Attribute | Value |
|---|---|
| `id` | `e_shorts_max_10_words_per_frame` |
| `name` | Max 10 Words Per Frame |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Every frame with visible on-screen text contains no more than 10 words.<br>Frames without visible text pass this criterion.</code></pre> |
| `prompt_template` | <pre><code>Does every frame with visible on-screen text contain 10 words or fewer?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and count visible on-screen words per frame.<br><br>3. Treat readable brand names, subtitles, captions, supers, stickers, disclaimers, and product text as visible words.<br><br>4. Return False if any single frame contains more than 10 visible words.<br><br>5. Provide the timestamp or frame evidence for the highest word count observed. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_production_style_index` - Production Style

| Attribute | Value |
|---|---|
| `id` | `shorts_production_style_index` |
| `name` | Production Style |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies the visual 'Lo-Fi' vs. 'Hi-Fi' characteristics of the video. <br>Measures the presence of UGC (User Generated Content) markers such as <br>handheld camera movement, natural lighting, and native mobile aesthetics. <br>Assesses if the video feels like an 'organic post' or a 'produced commercial.'</code></pre> |
| `prompt_template` | <pre><code>Act as a professional Cinematographer and Social Media Strategist. Your goal is to <br>quantify the 'UGC Authenticity' of the production style for this video.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. METRIC DEFINITIONS:<br>- **Density Score:** (Duration of shots that appear native/UGC) / (Total video duration). <br>  Represented as density_score in the JSON.<br>- **Authenticity Rating:** (0.0 - 1.0) 1.0 = Indistinguishable from an organic user upload; <br>  0.5 = High-quality "Studio-UGC" (professional gear mimicking a mobile look); <br>  0.0 = Traditional high-budget glossy commercial.<br><br>### 2. PRODUCTION MARKERS:<br>- **UGC/Lo-Fi:** Visible handheld jitter, natural/ambient lighting, mobile sensor resolution, "face-to-lens" intimacy.<br>- **Studio-UGC:** Polished vertical framing, stabilized movement, crisp external-mic audio, but retaining a casual feel.<br>- **High-Production:** Cinema-grade lenses, shallow depth of field, artificial 3-point lighting, professional color grading.<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float, # Certainty of production style classification<br>    "feature_quality_score": float, # 0.0-1.0 (How well it executes the intended style)<br>    "metrics": {{<br>        "density_score": float, # MANDATORY: UGC-style duration / Total video duration<br>        "camera_stability": "Handheld" &#124; "Stabilized" &#124; "Tripod/Static",<br>        "lighting_type": "Natural/Ambient" &#124; "Studio-Polished" &#124; "Raw/Incidental",<br>        "equipment_look": "Mobile_Phone" &#124; "Professional_Camera" &#124; "Webcam"<br>    }},<br>    "aesthetic_analysis": {{<br>        "platform_native_elements": boolean, # Use of top social media app fonts/stickers<br>        "environment_realism": "Lived-in/Messy" &#124; "Curated/Clean" &#124; "Studio/Abstract",<br>        "interface_compliance": "Safe_Zone_Optimized" &#124; "UI_Overlapped"<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "style_type": "True_UGC" &#124; "Studio_UGC" &#124; "Corporate_Ad",<br>            "description": str <br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "lofi_index": float, # 0.0 (Glossy) to 1.0 (Raw)<br>        "is_hook_native_looking": boolean, # Does it look like an organic post in the first 2s?<br>        "summary": "Technical analysis of the production authenticity and stylistic approach"<br>    }}<br>}}<br><br>### EVALUATION LOGIC:<br>1. Observe camera physics: Mobile devices have distinct micro-jitters; cinema rigs have "weight."<br>2. Check for "Safe Zone" awareness: Is the subject framed to avoid being covered by top social media UI (avatars, descriptions)?<br>3. Calculate **density_score** by identifying the total runtime of "authentic-looking" footage.<br>4. Lower the **Authenticity Rating** if the video uses professional motion graphics or studio-exclusive color palettes.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_sfv_adaptation_high` - Short Form Video Adaptation

| Attribute | Value |
|---|---|
| `id` | `shorts_sfv_adaptation_high` |
| `name` | Short Form Video Adaptation |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Quantifies the 'Native Emulation' of the production. Measures how effectively <br>the video mimics organic social content through lo-fi aesthetics, handheld <br>camera physics, and non-commercial editing patterns.</code></pre> |
| `prompt_template` | <pre><code>Act as a Social Media Trends Analyst and Video Strategist. Your goal is to <br>quantify the 'UGC Authenticity' of the production style.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. PRODUCTION MARKERS:<br>- **Native/Lo-Fi:** Handheld jitter, natural ambient lighting, mobile sensor resolution, face-to-lens intimacy.<br>- **Studio-Hybrid:** Vertical framing and casual tone but with professional lighting or stabilized movement.<br>- **Commercial/Glossy:** Cinema lenses, 3-point lighting, professional color grading, or traditional ad pacing.<br><br>### 2. DENSITY &amp; QUALITY LOGIC:<br>- **Density Score:** (Duration of shots that appear native/organic) / (Total video duration).<br>- **Feature Quality Score (Authenticity):** <br>    * 0.9-1.0: Indistinguishable from an organic user post.<br>    * 0.7-0.8: High-quality Studio-UGC (mimics mobile but feels "clean").<br>    * 0.4-0.6: Polished commercial with minor native elements.<br>    * 0.0-0.3: Traditional high-budget commercial style.<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float, <br>    "feature_quality_score": float, # The 0.0-1.0 Authenticity Rating<br>    "metrics": {{<br>        "density_score": float, # Duration of organic style / Total duration<br>        "camera_stability": "Handheld" &#124; "Stabilized" &#124; "Static",<br>        "lighting_type": "Natural" &#124; "Studio" &#124; "Mixed",<br>        "edit_style": "Fast-Cut" &#124; "Long-Take" &#124; "Jump-Cuts"<br>    }},<br>    "aesthetic_analysis": {{<br>        "lofi_index": float, # 0.0 (Glossy) to 1.0 (Raw)<br>        "is_hook_native": boolean, # Does the first 3s look like a post, not an ad?<br>        "platform_native_vibe": boolean # Use of native-style fonts/graphics<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "style_type": "Native" &#124; "Hybrid" &#124; "Commercial",<br>            "description": str<br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "visual_impact_score": float, # Effectiveness for mobile engagement<br>        "summary": "Concise technical summary of production authenticity"<br>    }}<br>}}<br><br>### EVALUATION LOGIC:<br>- **The Hook:** If the first 3 seconds are "Native" in style, increase the feature_quality_score.<br>- **Intent:** Distinguish between intentional Lo-Fi style and poor production quality.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_emoji_usage` - Emoji Usage

| Attribute | Value |
|---|---|
| `id` | `shorts_emoji_usage` |
| `name` | Emoji Usage |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Detects intentional creative emoji use: 1. Standard characters in text, <br>2. Animated effects, 3. Emoji-style stickers/graphics, <br>4. Platform-specific features. Excludes incidental background captures.</code></pre> |
| `prompt_template` | <pre><code>Act as a Visual Researcher and Social Media Analyst. Your goal is to <br>identify and quantify the use of emojis as creative overlays.<br><br>VIDEO METADATA: {metadata_summary}<br><br>### 1. EMOJI TYPES:<br>- **Standard:** Unicode emoji characters within text overlays.<br>- **Animated:** Emojis that pop, shake, or move.<br>- **Stickers:** Large, graphical emoji-style elements or platform-native stickers.<br><br>### 2. DENSITY &amp; QUALITY LOGIC:<br>- **Density Score:** (Sum of seconds with visible emojis) / (Total video duration).<br>- **Feature Quality Score:** <br>    * 0.9-1.0: Strategic use (syncs with audio, emphasizes key points).<br>    * 0.6-0.8: Moderate use, primarily decorative.<br>    * 0.1-0.5: Incidental or poorly placed (covers key subjects).<br>    * 0.0: No emojis detected.<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "feature_quality_score": float, # 0.0-1.0 based on narrative relevance<br>    "metrics": {{<br>        "density_score": float, # Total emoji duration / Total duration<br>        "emoji_count_estimate": int, <br>        "style": "Static" &#124; "Animated" &#124; "Kinetic",<br>        "placement": "Anchored_to_Text" &#124; "Floating" &#124; "Center_Pop"<br>    }},<br>    "spatial_analysis": {{<br>        "safe_zone_compliance": boolean, # Avoids UI overlap at bottom/right<br>        "primary_purpose": "Emphasis" &#124; "Tone" &#124; "CTA" &#124; "Decorative"<br>    }},<br>    "temporal_segments": [<br>        {{<br>            "start": float,<br>            "end": float,<br>            "emoji_type": str,<br>            "description": str<br>        }}<br>    ],<br>    "overall_assessment": {{<br>        "is_hook_emoji": boolean, # Emoji present in the first 2 seconds?<br>        "visual_impact_score": float, # Contribution to "organic" feel<br>        "summary": "Summary of emoji integration and effectiveness"<br>    }}<br>}}<br><br>### EVALUATION LOGIC:<br>- **Placement:** Lower the quality score if emojis are in the "Dead Zone" (bottom/right where UI buttons sit).<br>- **Relevance:** Higher impact if the emoji matches the spoken word or emotional tone.</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_personal_character_talk` - Direct to Camera Character Talk

| Attribute | Value |
|---|---|
| `id` | `shorts_personal_character_talk` |
| `name` | Direct to Camera Character Talk |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Evaluates the intimacy and continuity of direct lens address. <br>Measures the 'Breaking of the Fourth Wall' through gaze and conversational delivery.</code></pre> |
| `prompt_template` | <pre><code>Act as a Cinematographer and Parasocial Interaction Specialist. <br>Evaluate: How effectively does the character connect with the viewer via direct lens address?<br><br>VIDEO METADATA: {metadata_summary}<br><br>SCORE ON THREE DIMENSIONS (0-100 each):<br><br>1. GAZE CONTINUITY &amp; INTENSITY (45% weight)<br>   Focus: Are the eyes locked on the lens? Is it a "FaceTime" feel?<br>   - 90-100: Constant, unwavering eye contact that feels personal.<br>   - 70-89: Frequent direct looks, clear address to viewer.<br>   - 0-49: Looking at self on screen or off-camera; incidental gaze.<br><br>2. DELIVERY INTIMACY (30% weight)<br>   Focus: Conversational proximity and vocal tone.<br>   - 90-100: Casual, peer-to-peer tone; feels unscripted and close.<br>   - 70-89: Clear address but slightly presentational.<br>   - 0-69: Corporate or formal delivery; distant.<br><br>3. TEMPORAL DOMINANCE (25% weight)<br>   Focus: How much of the narrative is led by this direct address?<br>   - 90-100: Character address drives the entire video narrative.<br>   - 70-89: Significant segments of direct address.<br>   - 0-69: Brief intro/outro address only.<br><br>FINAL CALCULATION:<br>Overall Score = (Gaze × 0.45) + (Intimacy × 0.30) + (Dominance × 0.25)<br><br>FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "evaluation": {{<br>        "character_type": str, # e.g., "Creator", "Mascot"<br>        "gaze_consistency": "high"&#124;"medium"&#124;"low",<br>        "delivery_style": str,<br>        "gaze_intensity_score": int,<br>        "delivery_intimacy_score": int,<br>        "temporal_dominance_score": int,<br>        "weighted_overall": float<br>    }},<br>    "metrics": {{<br>        "density_score": float, # Duration of direct address / Total duration<br>        "is_hook_direct": boolean<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_native_brand_context` - Brand Secondary Element

| Attribute | Value |
|---|---|
| `id` | `shorts_native_brand_context` |
| `name` | Brand Secondary Element |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Evaluates if the brand is positioned as a secondary, natural element. <br>High scores indicate the brand feels like part of the environment, not a forced ad.</code></pre> |
| `prompt_template` | <pre><code>Act as a Brand Integration Analyst. <br>Evaluate: Is the brand naturally secondary within the organic content?<br><br>VIDEO METADATA:<br>{metadata_summary}<br><br>SCORE ON THREE DIMENSIONS (0-100 each):<br><br>1. NARRATIVE INTEGRATION (40% weight)<br>   Focus: Does the brand exist as a natural prop or mention in a larger story?<br>   - 90-100: Brand is seamless; story makes sense even without the logo.<br>   - 70-89: Brand is clearly secondary but narrative-aligned.<br>   - 0-69: Narrative feels forced around the brand (Ad-like).<br><br>2. VISUAL SUBTLETY (35% weight)<br>   Focus: Is the brand positioned to avoid "Ad Blindness"?<br>   - 90-100: Brand is clear but not the focal point (e.g., on clothing or background).<br>   - 70-89: Brand is visible but doesn't dominate the center frame.<br>   - 0-69: Brand is center-frame, dominant, or high-contrast (Forced focus).<br><br>3. CONTEXTUAL RELEVANCE (25% weight)<br>   Focus: Does the brand fit the "Lived-in" environment?<br>   - 90-100: Perfectly fits the setting (e.g., gym brand in a gym).<br>   - 70-89: Logical fit, but feels slightly staged.<br>   - 0-69: Out of place; studio environment.<br><br>FINAL CALCULATION:<br>Overall Score = (Narrative × 0.40) + (Subtle × 0.35) + (Context × 0.25)<br><br>FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "evaluation": {{<br>        "integration_method": str, # e.g., "prop", "attire", "verbal"<br>        "primary_focus": str, # What is the main focus if not the brand?<br>        "narrative_score": int,<br>        "visual_subtle_score": int,<br>        "context_score": int,<br>        "weighted_overall": float<br>    }},<br>    "metrics": {{<br>        "brand_density": float,<br>        "is_brand_dominant": boolean # False is better for this feature<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_personal_character_type` - Everyday Persona Validation

| Attribute | Value |
|---|---|
| `id` | `shorts_personal_character_type` |
| `name` | Everyday Persona Validation |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Determines if the video is led by a relatable 'everyday person' or creator. <br>Returns negative if the character is a professional actor, celebrity, <br>or fictional/animated entity.</code></pre> |
| `prompt_template` | <pre><code>Evaluate if the primary character in this ad is an 'Everyday Person'.<br><br>### DEFINITION:<br>An 'Everyday Person' is an organic creator or real user who feels <br>unpolished and relatable. This feature is FALSE if the person is a <br>professional actor, a famous celebrity, or a fictional character.<br><br>### FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean, # TRUE if Everyday Person/Creator, FALSE otherwise<br>    "confidence_score": float,<br>    "feature_quality_score": float, # 1.0 (Highly Authentic) to 0.0 (Clearly Commercial)<br>    "metrics": {{<br>        "is_everyday_person": boolean,<br>        "is_commercial_actor": boolean,<br>        "is_celebrity": boolean,<br>        "is_fictional_mascot": boolean<br>    }},<br>    "overall_assessment": {{<br>        "authenticity_rating": float, # 0.0 - 1.0 (How 'real' do they feel?)<br>        "sum": "Max 8 words identifying the person's vibe"<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_product_context` - Secondary Product Context

| Attribute | Value |
|---|---|
| `id` | `shorts_product_context` |
| `name` | Secondary Product Context |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Evaluates if the product is positioned as a secondary element rather than the main focus of the ad, <br>appearing in a natural and realistic context.</code></pre> |
| `prompt_template` | <pre><code>Act as a Product Stylist and Video Analyst. <br>Evaluate: Is the product used naturally as a secondary element in a realistic context?<br><br>VIDEO METADATA:<br>{metadata_summary}<br><br>SCORE ON THREE DIMENSIONS (0-100 each):<br><br>1. PRACTICAL UTILITY (45% weight)<br>   Focus: Is the product being used for its actual purpose naturally?<br>   - 90-100: Product usage is active but effortless (part of the background action).<br>   - 70-89: Usage is clear but slightly emphasized for the camera.<br>   - 0-69: Product is just "held" or static; trophy-like.<br><br>2. ENVIRONMENTAL REALISM (30% weight)<br>   Focus: Authenticity of the "Lived-in" space.<br>   - 90-100: Messy, real-world, or non-studio environment.<br>   - 70-89: Clean setting but clearly a home/office.<br>   - 0-69: Sterile white background or over-lit studio.<br><br>3. VISUAL WEIGHT (25% weight)<br>   Focus: Subject-to-Frame Ratio (SfR) balance.<br>   - 90-100: Product occupies &lt;20% of frame while in use.<br>   - 70-89: Product occupies 20-35% of frame.<br>   - 0-69: Product is dominant (&gt;50% of frame).<br><br>FINAL CALCULATION:<br>Overall Score = (Utility × 0.45) + (Realism × 0.30) + (Weight × 0.25)<br><br>FORMAT RESPONSE AS JSON:<br>{{<br>    "detected": boolean,<br>    "confidence_score": float,<br>    "evaluation": {{<br>        "usage_type": str,<br>        "environment_type": str,<br>        "utility_score": int,<br>        "realism_score": int,<br>        "visual_weight_score": int,<br>        "weighted_overall": float<br>    }},<br>    "metrics": {{<br>        "avg_sfr_percentage": float,<br>        "is_product_secondary": boolean<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `shorts_video_format` - Vertical Format Designed For Mobile

| Attribute | Value |
|---|---|
| `id` | `shorts_video_format` |
| `name` | Vertical Format Designed For Mobile |
| `category` | `SHORTS` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Verifies 9:16 portrait ratio and detects letterboxing/pillarboxing.</code></pre> |
| `prompt_template` | <pre><code>Verify if the video is optimized for mobile (9:16). <br><br>VIDEO METADATA:<br>{metadata_summary}<br><br>Metrics:<br>- feature_quality_score: 1.0 (True 9:16), 0.5 (Letterboxed/Square), 0.0 (Horizontal).<br><br>JSON ONLY:<br>{{<br>    "detected": bool,<br>    "confidence_score": float,<br>    "feature_quality_score": float, <br>    "metrics": {{<br>        "format": "9:16"&#124;"1:1"&#124;"16:9"&#124;"mixed",<br>        "is_letterboxed": bool,<br>        "safe_zone_compliant": bool<br>    }},<br>    "overall_assessment": {{<br>        "mobile_native_score": float,<br>        "sum": "Max 10 words summary"<br>    }}<br>}}</code></pre> |
| `extra_instructions` |  |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |
