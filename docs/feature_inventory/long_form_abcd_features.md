# Long Form ABCD Feature Definitions

Source: `features_repository/long_form_abcd_features.py`.

Each feature below includes every `VideoFeature` config attribute currently defined in code.

## Attract

### `a_dynamic_start` - Dynamic Start

| Attribute | Value |
|---|---|
| `id` | `a_dynamic_start` |
| `name` | Dynamic Start |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>The first shot in the video changes in less than 3 seconds.</code></pre> |
| `prompt_template` | <pre><code>Does the first shot in the video change in less than 3 seconds?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}.<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp when the first shot in the video changes.<br><br>4. Return True if and only if the first shot in the video changes in less than 3 seconds. |
| `evaluation_method` | `ANNOTATIONS` |
| `evaluation_function` | `detect_dynamic_start` |
| `include_in_evaluation` | `True` |
| `group_by` | `NO_GROUPING` |

### `a_quick_pacing` - Quick Pacing

| Attribute | Value |
|---|---|
| `id` | `a_quick_pacing` |
| `name` | Quick Pacing |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Within ANY 5 consecutive seconds there are 5 or more shots in the video. These include hard cuts, soft<br>transitions and camera changes such as camera pans, swipes, zooms, depth of field changes, tracking shots<br>and movement of the camera.</code></pre> |
| `prompt_template` | <pre><code>Are there 5 or more shots within ANY 5 consecutive seconds in the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the shot changes count in the following format: Number of shots: #<br><br>4. Provide the exact timestamp when the shot changes happen and a description of the shot.<br><br>5. Return False if and only if the number of identified shots is less than 5. |
| `evaluation_method` | `ANNOTATIONS` |
| `evaluation_function` | `detect_quick_pacing` |
| `include_in_evaluation` | `True` |
| `group_by` | `NO_GROUPING` |

### `a_quick_pacing_1st_5_secs` - Quick Pacing (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `a_quick_pacing_1st_5_secs` |
| `name` | Quick Pacing (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>There are at least 5 shot changes or visual cuts detected in the first 5 seconds of the video. These include hard cuts,<br>soft transitions and camera changes such as camera pans, swipes, zooms, depth of field changes,<br>tracking shots and movement of the camera.</code></pre> |
| `prompt_template` | <pre><code>Are there at least 5 shot changes or visual cuts detected in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the shot changes count in the following format: Number of shots: #<br><br>4. Provide the exact timestamp when the shot changes happen and a description of the shot.<br><br>5. Return False if the number of shots identified is less than 5. |
| `evaluation_method` | `ANNOTATIONS` |
| `evaluation_function` | `detect_quick_pacing_1st_5_secs` |
| `include_in_evaluation` | `True` |
| `group_by` | `NO_GROUPING` |

### `a_supers` - Supers

| Attribute | Value |
|---|---|
| `id` | `a_supers` |
| `name` | Supers |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Any supers (text overlays) have been incorporated at any time in the video.</code></pre> |
| `prompt_template` | <pre><code>Are there any supers (text overlays) at any time in the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp where supers are found as well as the list of supers. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `a_supers_with_audio` - Supers with Audio

| Attribute | Value |
|---|---|
| `id` | `a_supers_with_audio` |
| `name` | Supers with Audio |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `ATTRACT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>The speech heard in the audio of the video matches OR is contextually supportive of the overlaid<br>text shown on screen.</code></pre> |
| `prompt_template` | <pre><code>Does the speech match any supers (text overlays) in the video or is the speech contextually supportive<br>of the overlaid text shown on the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp where supers are found and the timestamp when<br>the speech matches the supers or is contextually supportive of the overlaid text shown on the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

## Brand

### `b_brand_mention_speech` - Brand Mention (Speech)

| Attribute | Value |
|---|---|
| `id` | `b_brand_mention_speech` |
| `name` | Brand Mention (Speech) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>The brand name is heard in the audio or speech at any time in the video.</code></pre> |
| `prompt_template` | <pre><code>Does the speech mention the brand {brand_name} at any time in the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Provide the exact timestamp when the brand {brand_name} is heard in the speech of the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `b_brand_mention_speech_1st_5_secs` - Brand Mention (Speech) (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `b_brand_mention_speech_1st_5_secs` |
| `name` | Brand Mention (Speech) (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>The brand name is heard in the audio or speech in the first 5 seconds of the video.</code></pre> |
| `prompt_template` | <pre><code>Does the speech mention the brand {brand_name} in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Provide the exact timestamp when the brand {brand_name} is heard in the speech of the video.<br><br>3. Return True if and only if the brand {brand_name} is heard in the speech of the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

### `b_brand_visuals` - Brand Visuals

| Attribute | Value |
|---|---|
| `id` | `b_brand_visuals` |
| `name` | Brand Visuals |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Branding, defined as the brand name or brand logo are shown in-situation or overlaid at any time in the video.</code></pre> |
| `prompt_template` | <pre><code>Is the brand {brand_name} or brand logo {brand_name} visible at any time in the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp when the brand {brand_name} or brand logo {brand_name} is found. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `b_brand_visuals_1st_5_secs` - Brand Visuals (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `b_brand_visuals_1st_5_secs` |
| `name` | Brand Visuals (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>Branding, defined as the brand name or brand logo are shown in-situation or overlaid in the first 5 seconds of the video.</code></pre> |
| `prompt_template` | <pre><code>Is the brand {brand_name} or brand logo {brand_name} visible in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp when the brand {brand_name} or brand logo {brand_name} is found. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

### `b_product_mention_speech` - Product Mention (Speech)

| Attribute | Value |
|---|---|
| `id` | `b_product_mention_speech` |
| `name` | Product Mention (Speech) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>The branded product names or generic product categories are heard or mentioned in the audio or speech<br>at any time in the video.</code></pre> |
| `prompt_template` | <pre><code>Are any of the following products: {branded_products} or product categories: {branded_products_categories}<br>heard at any time in the speech of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Provide the exact timestamp when the products {branded_products} or product categories<br>{branded_products_categories} are heard in the speech of the video.<br><br>3. Return False if the products or product categories are not heard in the speech.<br><br>4. Only strictly use the speech of the video to answer, don't consider visual elements. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `b_product_mention_speech_1st_5_secs` - Product Mention (Speech) (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `b_product_mention_speech_1st_5_secs` |
| `name` | Product Mention (Speech) (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>The branded product names or generic product categories are heard or mentioned in the audio or speech<br>in the first 5 seconds of the video.</code></pre> |
| `prompt_template` | <pre><code>Are any of the following products: {branded_products} or product categories: {branded_products_categories}<br>heard in the speech in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Provide the exact timestamp when the products {branded_products}<br>or product categories {branded_products_categories} are heard in the speech of the video.<br><br>3. Return False if the products or product categories are not heard in the speech.<br><br>4. Only strictly use the speech of the video to answer, don't consider visual elements. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

### `b_product_mention_text` - Product Mention (Text)

| Attribute | Value |
|---|---|
| `id` | `b_product_mention_text` |
| `name` | Product Mention (Text) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>The branded product names or generic product categories are present in any text or overlay at any<br>time in the video.</code></pre> |
| `prompt_template` | <pre><code>Is any of the following products: {branded_products} or product categories: {branded_products_categories}<br>present in any text or overlay at any time in the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Provide the exact timestamp when the products {branded_products} or product categories:<br>{branded_products_categories} are found  in any text or overlay in the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `b_product_mention_text_1st_5_secs` - Product Mention (Text) (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `b_product_mention_text_1st_5_secs` |
| `name` | Product Mention (Text) (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>The branded product names or generic product categories are present in any text or overlay in the first 5 seconds of the video.</code></pre> |
| `prompt_template` | <pre><code>Is any of the following products: {branded_products} or product categories: {branded_products_categories}<br>present in any text or overlay in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Provide the exact timestamp when the products {branded_products} or product categories: {branded_products_categories} are found in any text or overlay in the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

### `b_product_visuals` - Product Visuals

| Attribute | Value |
|---|---|
| `id` | `b_product_visuals` |
| `name` | Product Visuals |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>A product or branded packaging is visually present at any time in the video. Where the product is a service a relevant<br>substitute should be shown such as via a branded app or branded service personnel.</code></pre> |
| `prompt_template` | <pre><code>Is any of the following products: {branded_products} or product categories: {branded_products_categories}<br>visually present at any time in the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Provide the exact timestamp when the products {branded_products} or product categories:<br>{branded_products_categories} are visually present.<br><br>3. Return True if and only if the branded products or product categories are visually present in the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `b_product_visuals_1st_5_secs` - Product Visuals (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `b_product_visuals_1st_5_secs` |
| `name` | Product Visuals (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `BRAND` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>A product or branded packaging is visually present in the first 5 seconds of the video. Where the product is a service,<br>a relevant substitute should be shown such as via a branded app or branded service personnel.</code></pre> |
| `prompt_template` | <pre><code>Is any of the following products: {branded_products} or product categories: {branded_products_categories}<br>visually present in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Provide the exact timestamp when the products {branded_products} or product categories:<br>{branded_products_categories} are visually present.<br><br>3. Return True if and only if the branded products or product categories are visually present in the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

## Connect

### `c_overall_pacing` - Overall Pacing

| Attribute | Value |
|---|---|
| `id` | `c_overall_pacing` |
| `name` | Overall Pacing |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>The pace of the video is less than 2 seconds per shot/frame.</code></pre> |
| `prompt_template` | <pre><code>Is the pace of video less than 2 seconds per shot/frame?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Return True if and only if the pace of video less than 2 seconds per shot/frame. |
| `evaluation_method` | `ANNOTATIONS` |
| `evaluation_function` | `detect_overall_pacing` |
| `include_in_evaluation` | `True` |
| `group_by` | `NO_GROUPING` |

### `c_presence_of_people` - Presence of People

| Attribute | Value |
|---|---|
| `id` | `c_presence_of_people` |
| `name` | Presence of People |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>People are shown in any capacity at any time in the video. Any human body parts are acceptable to pass<br>this guideline. Alternate representations of people such as Animations or Cartoons ARE acceptable.</code></pre> |
| `prompt_template` | <pre><code>Are there people present at any time in the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp when people are present in the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `c_presence_of_people_1st_5_secs` - Presence of People (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `c_presence_of_people_1st_5_secs` |
| `name` | Presence of People (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>People are shown in any capacity in the first 5 seconds of the video. Any human body parts are acceptable to pass this guideline.<br>Alternate representations of people such as Animations or Cartoons ARE acceptable.</code></pre> |
| `prompt_template` | <pre><code>Are there people present in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp when people are present in the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

### `c_visible_face` - Visible Face (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `c_visible_face` |
| `name` | Visible Face (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>At least one human face is present in the first 5 seconds of the video. Alternate representations of people such as<br>Animations or Cartoons ARE acceptable.</code></pre> |
| `prompt_template` | <pre><code>Is there a human face present in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp when the human face is present. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

### `c_visible_face_close_up` - Visible Face (Close Up)

| Attribute | Value |
|---|---|
| `id` | `c_visible_face_close_up` |
| `name` | Visible Face (Close Up) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `CONNECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>There is a close up of a human face at any time in the video.</code></pre> |
| `prompt_template` | <pre><code>Is there a close up of a human face present at any time the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and answer the question.<br><br>3. Provide the exact timestamp when there is a close up of a human face. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

## Direct

### `d_audio_speech_early_1st_5_secs` - Audio Early (First 5 seconds)

| Attribute | Value |
|---|---|
| `id` | `d_audio_speech_early_1st_5_secs` |
| `name` | Audio Early (First 5 seconds) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `DIRECT` |
| `video_segment` | `FIRST_5_SECS_VIDEO` |
| `evaluation_criteria` | <pre><code>Speech is detected in the audio in the first 5 seconds of the video.</code></pre> |
| `prompt_template` | <pre><code>Is speech detected in the audio in the first 5 seconds of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Only strictly use the speech of the video to answer. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FIRST_5_SECS_VIDEO` |

### `d_call_to_action_speech` - Call To Action (Speech)

| Attribute | Value |
|---|---|
| `id` | `d_call_to_action_speech` |
| `name` | Call To Action (Speech) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `DIRECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>A 'Call To Action' phrase is heard or mentioned in the audio or speech at any time in the video.</code></pre> |
| `prompt_template` | <pre><code>Is any call to action heard or mentioned in the speech of the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Some examples of call to actions are: {call_to_actions}<br><br>3. Provide the exact timestamp when the call to actions are heard or mentioned in the speech of the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

### `d_call_to_action_text` - Call To Action (Text)

| Attribute | Value |
|---|---|
| `id` | `d_call_to_action_text` |
| `name` | Call To Action (Text) |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `DIRECT` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>A 'Call To Action' phrase is detected in the video supers (overlaid text) at any time in the video.</code></pre> |
| `prompt_template` | <pre><code>Is any call to action detected in any text overlay at any time in the video?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Some examples of call to actions are: {call_to_actions}<br><br>3. Look through each frame in the video carefully and answer the question.<br><br>4. Provide the exact timestamp when the call to action is detected in any text overlay in the video. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |

## Other / None

### `e_long_form_max_10_words_per_frame` - Max 10 Words Per Frame

| Attribute | Value |
|---|---|
| `id` | `e_long_form_max_10_words_per_frame` |
| `name` | Max 10 Words Per Frame |
| `category` | `LONG_FORM_ABCD` |
| `sub_category` | `NONE` |
| `video_segment` | `FULL_VIDEO` |
| `evaluation_criteria` | <pre><code>Every frame with visible on-screen text contains no more than 10 words.<br>Frames without visible text pass this criterion.</code></pre> |
| `prompt_template` | <pre><code>Does every frame with visible on-screen text contain 10 words or fewer?</code></pre> |
| `extra_instructions` | 1. Consider the following criteria for your answer: {criteria}<br><br>2. Look through each frame in the video carefully and count visible on-screen words per frame.<br><br>3. Treat readable brand names, subtitles, captions, supers, disclaimers, and product text as visible words.<br><br>4. Return False if any single frame contains more than 10 visible words.<br><br>5. Provide the timestamp or frame evidence for the highest word count observed. |
| `evaluation_method` | `LLMS` |
| `evaluation_function` | `` |
| `include_in_evaluation` | `True` |
| `group_by` | `FULL_VIDEO` |
