class ClipTextEncodeCombined:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                    }),
                "negative_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                    }),
                },
            }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")

    FUNCTION = "execute"

    OUTPUT_NODE = False

    CATEGORY = "Aronas Nodes"

    def execute(self, clip, positive_text, negative_text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        pos_tokens = clip.tokenize(positive_text)
        neg_tokens = clip.tokenize(negative_text)

        return (clip.encode_from_tokens_scheduled(pos_tokens),
                clip.encode_from_tokens_scheduled(neg_tokens))

NODE_CLASS_MAPPINGS = {"Node": ClipTextEncodeCombined}

NODE_DISPLAY_NAME_MAPPINGS = {"Node": "CLIP Text Encode (Combined Prompt)"}