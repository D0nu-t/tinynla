class SemanticLabeler:

    def __init__(self):
        pass

    def describe(self, text: str) -> str:

        text_lower = text.lower()

        entities = []

        emotions = []

        themes = []

        # -----------------------------------
        # Entities
        # -----------------------------------

        entity_words = {
            "dog": "dog",
            "cat": "cat",
            "boy": "boy",
            "girl": "girl",
            "king": "king",
            "queen": "queen",
            "dragon": "dragon",
            "robot": "robot",
            "teacher": "teacher",
            "mother": "mother"
        }

        for word, label in entity_words.items():

            if word in text_lower:
                entities.append(label)

        # -----------------------------------
        # Emotions
        # -----------------------------------

        emotion_words = {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "afraid": "fearful",
            "excited": "excited"
        }

        for word, label in emotion_words.items():

            if word in text_lower:
                emotions.append(label)

        # -----------------------------------
        # Themes
        # -----------------------------------

        theme_words = {
            "school": "education",
            "magic": "fantasy",
            "space": "science fiction",
            "forest": "nature",
            "castle": "royalty"
        }

        for word, label in theme_words.items():

            if word in text_lower:
                themes.append(label)

        # -----------------------------------
        # Compose Description
        # -----------------------------------

        parts = []

        if entities:

            parts.append(
                "Entities: "
                + ", ".join(sorted(set(entities)))
            )

        if emotions:

            parts.append(
                "Emotions: "
                + ", ".join(sorted(set(emotions)))
            )

        if themes:

            parts.append(
                "Themes: "
                + ", ".join(sorted(set(themes)))
            )

        if len(parts) == 0:

            return (
                "General narrative text."
            )

        return " | ".join(parts)